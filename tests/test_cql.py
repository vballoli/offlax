import d4rl
import gym
import jax
from jax import numpy as jnp

from offlax.cql import CQLDiscrete
from offlax.models import ActorDiscrete, Critic


def test_cql():
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    env = gym.make("maze2d-open-v0")

    actor = ActorDiscrete(64, env.action_space.shape[0])

    critic = Critic(64, env.action_space.shape[0])

    state = jnp.ones((10, env.observation_space.shape[0]))

    cql = CQLDiscrete(
        rng,
        actor,
        critic,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        0.1,
        0.1,
    )

    action, probs, logs = cql.get_action(state, rng=rng, train=True)

    experience_batch = [
        jnp.ones((128, env.observation_space.shape[0])),
        jnp.ones((128, 1)),
        jnp.ones((128, 1)),
        jnp.ones((128, env.observation_space.shape[0])),
        jnp.ones((128, 1)),
    ]
    cql.step(experience_batch)
