from offlax.models import Actor, Critic

import jax
from jax import numpy as jnp
from omegaconf import OmegaConf


def test_actor():
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    actor = Actor(64, 64)
    actor_variables = actor.init(init_rng, jnp.ones((1, 20)))

    state = jnp.ones((1, 20))
    mu, sigma = actor.apply(actor_variables, state)

    a = actor.get_action(actor_variables, state, rng)
    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)

    config_dict = actor.get_config_dict()
    assert config_dict['hidden_dim'] == 64
    assert config_dict['output_dim'] == 64

def test_critic():
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    critic = Critic(64, 64)
    critic_variables = critic.init(init_rng, jnp.ones((1, 20)))

    state = jnp.ones((1, 20))
    action = critic.apply(critic_variables, state)
    assert isinstance(action, jnp.ndarray)

    config_dict = critic.get_config_dict()
    assert config_dict['hidden_dim'] == 64
    assert config_dict['output_dim'] == 64