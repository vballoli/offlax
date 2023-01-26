from copy import deepcopy
from typing import List, Tuple

import jax
import optax
from jax import numpy as jnp

from offlax.models import Actor, Critic


class CQL:
    """Implementation of Conservative Q Learning (CQL) algorithm.

    Paper: https://arxiv.org/abs/2006.04779
    """

    def __init__(
        self,
        rng: jax.random.PRNGKey,
        actor: Actor,
        critic: Critic,
        state_dims: List[int],
        action_dims: int,
        gamma: float,
        tau: float,
    ):
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
        self.critic_target1_variables = deepcopy(self.critic_target1)

        self.critic2 = critic
        self.critic2_variables = self.critic2.init(rng, jnp.ones((1, state_dims)))
        self.critic2_optimizer = optax.adam(1e-3)
        self.critic2_optimizer_variables = self.critic2_optimizer.init(
            self.critic2_variables
        )

        self.critic_target2 = deepcopy(critic)
        self.critic_target2_variables = deepcopy(self.critic_target2)

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

        return action

    def get_actor_loss(
        self,
        states: jnp.ndarray,
        actor_variables,
        critic1_variables,
        critic2_variables,
        alpha: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        actions, log_preds_actions = self.actor.apply(actor_variables, states)

        q1 = self.critic1.apply(critic1_variables, states)
        q2 = self.critic2.apply(critic2_variables, states)

        min_q = jnp.minimum(q1, q2)
        actor_loss = jnp.mean((alpha * log_preds_actions - min_q))

        return actor_loss, log_preds_actions

    def get_alpha_loss(self, alpha: jnp.ndarray, log_preds: jnp.ndarray) -> jnp.ndarray:
        return -(alpha * jax.lax.stop_gradient(log_preds + self.target_entropy)).mean()

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
