import jax
from jax import numpy as jnp

from offlax.cql import CQLDiscrete
from offlax.models import ActorDiscrete, Critic


def test_cql():
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    actor = ActorDiscrete(64, 3)

    critic = Critic(64, 1)

    state = jnp.ones((10, 20))

    cql = CQLDiscrete(rng, actor, critic, 20, 3, 0.1, 0.1)

    action, probs, logs = cql.get_action(state, rng=rng, train=True)
    # assert False, probs.shape

    experience_batch = [
        jnp.ones((10, 20)),
        jnp.ones((10, 1)),
        jnp.ones((10, 1)),
        jnp.ones((10, 20)),
        jnp.ones((10, 1)),
    ]
    cql.step(experience_batch)
