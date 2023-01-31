import d4rl
import gym
import jax
import warnings

warnings.filterwarnings("ignore")
from jax import numpy as jnp


from offlax.cql import CQLDiscrete
from offlax.models import ActorDiscrete, Critic
from offlax.runner import OfflaxRunner


def test_runner_search():
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    env = gym.make("maze2d-open-v0")
    actor = ActorDiscrete(64, env.action_space.shape[0])

    critic = Critic(64, 1)

    cql = CQLDiscrete(
        rng,
        actor,
        critic,
        env.observation_space.shape[0],
        env.action_space.shape[0],
        0.1,
        0.1,
    )

    runner = OfflaxRunner(cql, enable_wandb=False)

    runner.train()
