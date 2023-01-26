from offlax.models import Actor, Critic
from offlax.cql import CQL

import jax
from jax import numpy as jnp


def test_cql():
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    actor = Actor(64, 3)
    
    critic = Critic(64, 1)
    
    state = jnp.ones((1, 20))

    cql = CQL(rng, actor, critic, 20, 3, 0.1, 0.1)

    cql.get_action(state, rng=rng)

    experience_batch = [jnp.ones((10, 20)), jnp.ones((10, 1)), jnp.ones((10, 1)), jnp.ones((10, 20)), jnp.ones((10, 1))]
    cql.step(experience_batch)