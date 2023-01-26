from copy import deepcopy
from typing import List, Tuple

import jax
import optax
from jax import numpy as jnp
from flax import linen as nn

from offlax.models import ActorDiscrete, ActorContinuous, Critic


class CQLDiscrete:
    """Implementation of Conservative Q Learning (CQL) algorithm.

    Paper: https://arxiv.org/abs/2006.04779
    """

    def __init__(
        self,
        rng: jax.random.PRNGKey,
        actor: ActorDiscrete,
        critic: Critic,
        state_dims: List[int],
        action_dims: int,
        gamma: float,
        tau: float,
    ):
        self.rng = rng
        self.actor = actor
        self.actor_variables = self.actor.init(rng, jnp.ones((1, state_dims)))
        self.actor_optimizer = optax.adam(1e-3)
        self.actor_optimizer_variables = self.actor_optimizer.init(self.actor_variables)

        self.critic1 = critic
        self.critic1_variables = self.critic1.init(rng, jnp.ones((1, state_dims)))
        self.critic1_optimizer = optax.adam(1e-3)
        self.critic1_optimizer_variables = self.critic1_optimizer.init(
            self.critic1_variables
        )

        self.critic_target1 = deepcopy(critic)
        self.critic_target1_variables = deepcopy(self.critic1_variables)

        self.critic2 = critic
        self.critic2_variables = self.critic2.init(rng, jnp.ones((1, state_dims)))
        self.critic2_optimizer = optax.adam(1e-3)
        self.critic2_optimizer_variables = self.critic2_optimizer.init(
            self.critic2_variables
        )

        self.critic_target2 = deepcopy(critic)
        self.critic_target2_variables = deepcopy(self.critic2_variables)

        self.alpha = jnp.zeros((1))
        self.alpha_optimizer = optax.adam(1e-3)
        self.alpha_optimizer_variables = self.alpha_optimizer.init(self.alpha)

        self.target_entropy = -float(action_dims)

        self.gamma = gamma
        self.tau = tau

    def get_action(
        self, state: jnp.ndarray, train: bool = False, rng: jax.random.PRNGKey = None
    ) -> jnp.ndarray:
        state = jax.lax.stop_gradient(state)

        assert rng is not None

        action = self.actor.get_action(
            self.actor_variables, state, deterministic=not train, key=rng
        )

        action = jax.lax.stop_gradient(action)

        return action

    def get_actor_loss(
        self,
        states: jnp.ndarray,
        actor_variables,
        critic1_variables,
        critic2_variables,
        alpha: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        self.rng, loss_key = jax.random.split(self.rng)
        actions, action_probabilities, log_preds_actions = self.actor.get_action(actor_variables, states, loss_key, return_log_prob=True)

        q1 = self.critic1.apply(critic1_variables, states)
        q2 = self.critic2.apply(critic2_variables, states)

        min_q = jnp.minimum(q1, q2)
        actor_loss = jnp.mean(jnp.sum(action_probabilities * (alpha * log_preds_actions - min_q), axis=1))
        log_action_sum = jnp.sum(log_preds_actions * action_probabilities)

        return actor_loss, log_action_sum

    def get_alpha_loss(self, alpha: jnp.ndarray, log_preds: jnp.ndarray) -> jnp.ndarray:
        return -(alpha * jax.lax.stop_gradient(log_preds + self.target_entropy)).mean()

    def get_critic_loss(self, states: jnp.ndarray, next_states: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray, critic1_variables, critic2_variables):
        self.rng, actor_key = jax.random.split(self.rng)
        action, action_probs, log_prob_sum = self.actor.get_action(self.actor_variables, next_states, actor_key, deterministic=False, return_log_prob=True)

        q_target_1_next = jax.lax.stop_gradient(self.critic_target1.apply(self.critic_target1_variables, next_states))
        q_target_2_next = jax.lax.stop_gradient(self.critic_target2.apply(self.critic_target2_variables, next_states))

        q_target_next = jax.lax.stop_gradient(action_probs * (jnp.minimum(q_target_1_next, q_target_2_next) - self.alpha * log_prob_sum))

        q_targets = jax.lax.stop_gradient(rewards + (self.gamma * (1 - dones) * jnp.expand_dims(q_target_next.sum(1), 1)))

        q1 = self.critic1.apply(critic1_variables, states)
        q2 = self.critic2.apply(critic2_variables, states)

        q1_ = jnp.take_along_axis(q1, action.astype('long'), 1)
        q2_ = jnp.take_along_axis(q2, action.astype('long'), 1)

        critic1_loss = 0.5 * jnp.square(q1_-q_targets)
        critic2_loss = 0.5 * jnp.square(q2_-q_targets)

        cql1_scaled_loss = jnp.log(jnp.sum(jnp.exp(q1), 1))
        cql2_scaled_loss = jnp.log(jnp.sum(jnp.exp(q2), 1))

        total_c1_loss = (critic1_loss).sum() + (cql1_scaled_loss).sum()
        total_c2_loss = (critic2_loss).sum() + (cql2_scaled_loss).sum()

        return total_c1_loss, total_c2_loss

    def step(self, experience_batch: List):
        states, actions, rewards, next_states, dones = experience_batch

        # Calculate Actor loss and update actor variables
        alpha = deepcopy(self.alpha)
        (actor_loss, log_preds_actor), actor_gradients = jax.value_and_grad(
            self.get_actor_loss, has_aux=True, argnums=1
        )(
            states,
            self.actor_variables,
            self.critic1_variables,
            self.critic2_variables,
            alpha,
        )
        actor_updates, self.actor_optimizer_variables = self.actor_optimizer.update(
            actor_gradients, self.actor_optimizer_variables
        )
        self.actor_variables = optax.apply_updates(self.actor_variables, actor_updates)

        # Update alpha
        alpha_loss = -(
            self.alpha * jax.lax.stop_gradient(log_preds_actor + self.target_entropy)
        ).mean()
        alpha_loss, alpha_gradients = jax.value_and_grad(self.get_alpha_loss)(
            self.alpha, log_preds_actor
        )
        alpha_updates, self.alpha_optimizer_variables = self.alpha_optimizer.update(
            alpha_gradients, self.alpha_optimizer_variables, self.alpha
        )
        self.alpha = optax.apply_updates(self.alpha, alpha_updates)

        # Update critics
        (total_c1_loss, total_c2_loss), (critic1_gradients, critic2_gradients)= jax.value_and_grad(
            self.get_critic_loss, has_aux=True, argnums=[4,5]
        )(states, next_states, rewards, dones, self.critic1_variables, self.critic2_variables)

        critic1_updates, self.critic1_optimizer_variables = self.critic1_optimizer.update(
            critic1_gradients, self.critic1_optimizer_variables, self.critic1_variables
        )
        self.critic1_variables = optax.apply_updates(self.critic1_variables, critic1_updates)

        critic2_updates, self.critic2_optimizer_variables = self.critic2_optimizer.update(
            critic2_gradients, self.critic2_optimizer_variables, self.critic2_variables
        )
        self.critic2_variables = optax.apply_updates(self.critic2_variables, critic2_updates)

        # Update target critics
        self.critic_target1_variables = jax.tree_map(lambda p, target_p: p * self.tau + target_p * (1 - self.tau), self.critic1_variables, self.critic_target1_variables)
        self.critic_target2_variables = jax.tree_map(lambda p, target_p: p * self.tau + target_p * (1 - self.tau), self.critic2_variables, self.critic_target2_variables)

        