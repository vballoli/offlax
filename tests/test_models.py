from typing import Tuple
from offlax.models import ActorDiscrete, ActorContinuous, Critic

import jax
from jax import numpy as jnp
from omegaconf import OmegaConf


def test_actor_continuous():
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    actor = ActorContinuous(64, 64)
    actor_variables = actor.init(init_rng, jnp.ones((1, 20)))

    state = jnp.ones((1, 20))
    mu, sigma = actor.apply(actor_variables, state)

    a = actor.get_action(actor_variables, state, rng)
    assert isinstance(mu, jnp.ndarray)
    assert isinstance(sigma, jnp.ndarray)

    config_dict = actor.get_config_dict()
    assert config_dict['hidden_dim'] == 64
    assert config_dict['output_dim'] == 64

def test_actor_discrete():
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    actor = ActorDiscrete(64, 64)
    actor_variables = actor.init(init_rng, jnp.ones((1, 20)))

    state = jnp.ones((1, 20))
    action = actor.apply(actor_variables, state)

    action = actor.get_action(actor_variables, state, rng, return_log_prob=True)
    assert isinstance(action, Tuple)
    assert len(action) == 3
    # assert isinstance(mu, jnp.ndarray)
    # assert isinstance(sigma, jnp.ndarray)

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