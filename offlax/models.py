from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple, Union

import jax
from flax import linen as nn
from flax.core.scope import VariableDict
from jax import numpy as jnp
from omegaconf import OmegaConf
from tensorflow_probability.substrates import jax as tfp
from ray import tune

tfd = tfp.distributions


class Policy(nn.Module):
    """Generic implementation of a Policy class"""

    @abstractmethod
    def get_action(self, variables: VariableDict, state: jnp.ndarray, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_config_dict(self) -> OmegaConf:
        raise NotImplementedError

    def save_config_dict(self, *args, **kwargs) -> None:
        config = self.get_config_dict()
        return config.save(*args, **kwargs)


class ActorContinuous(Policy):
    """Actor for a continuous action space"""

    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.relu(nn.Dense(self.hidden_dim, name=f"linear1_{self.__repr__()}")(x))
        x = nn.relu(nn.Dense(self.hidden_dim, name=f"linear2_{self.__repr__()}")(x))
        x = nn.relu(nn.Dense(self.hidden_dim, name=f"linear3_{self.__repr__()}")(x))

        mu = nn.Dense(self.output_dim, name=f"linear_mu_{self.__repr__()}")(x)
        log_std = nn.Dense(self.output_dim, name=f"linear_log_std_{self.__repr__()}")(x)

        return mu, log_std

    def get_action(
        self,
        variables: VariableDict,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        deterministic: bool = False,
        return_log_prob: bool = True,
    ) -> jnp.ndarray:
        """Returns action for a given state

        Args:
            variables (VariableDict): weights of the actor
            state (jnp.ndarray): state of the environment
            key (jax.random.PRNGKey): JAX random key
            deterministic (bool, optional): flag when True, return a deterministic action. Defaults to False.
            return_log_prob (bool, optional): flag when True, returns the probability and the log of the probability of the actor actions. Defaults to True.

        Returns:
            jnp.ndarray: _description_ #TODO: Update this
        """
        mu, log_std = self.apply(variables, state)

        if deterministic:
            mu = jax.lax.stop_gradient(mu)
            return jnp.tanh(mu)

        # normal distribution
        normal_dist = tfd.Normal(mu, jnp.exp(log_std))
        normal_dist_sample = normal_dist.sample(seed=key)
        action = jnp.tanh(normal_dist_sample)

        if return_log_prob:
            log_prob = normal_dist.log_prob(normal_dist_sample) - jnp.log(
                1 - action**2 + 1e-6
            ).sum(1, keepdims=True)
            return action, log_prob

        return action

    def get_config_dict(self, name: str = "") -> OmegaConf:
        config = OmegaConf.create({})
        if name != "":
            name += "/"
        config[f"{name}hidden_dim"] = self.hidden_dim
        config[f"{name}output_dim"] = self.output_dim

        if name == "":
            return config
        return {name: config}

    def get_search_space(self, prefix: str = ""):
        if prefix != "":
            prefix += "/"
        config = {f"{prefix}hidden_dim": tune.grid_search([32, 64, 128, 256])}
        return config


class ActorDiscrete(Policy):
    """Actor for a continuous action space"""

    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(nn.Dense(self.hidden_dim, name=f"linear1_{self.__repr__()}")(x))
        x = nn.relu(nn.Dense(self.hidden_dim, name=f"linear2_{self.__repr__()}")(x))
        x = nn.Dense(self.output_dim, name=f"linear3_{self.__repr__()}")(x)

        return nn.softmax(x)

    def get_action(
        self,
        variables: VariableDict,
        state: jnp.ndarray,
        key: jax.random.PRNGKey,
        deterministic: bool = False,
        return_log_prob: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns action for a given state

        Args:
            variables (VariableDict): weights of the actor
            state (jnp.ndarray): state of the environment
            key (jax.random.PRNGKey): JAX random key
            deterministic (bool, optional): flag when True, return a deterministic action. Defaults to False.
            return_log_prob (bool, optional): flag when True, returns the probability and the log of the probability of the actor actions. Defaults to True.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: _description_ #TODO: Update this
        """
        action_probability = self.apply(variables, state)
        categorical_distribution = tfd.Categorical(action_probability)

        action = categorical_distribution.sample(seed=key)

        if deterministic:
            return action

        if return_log_prob:
            log_action_probability = jnp.log(
                jnp.asarray(action_probability == 0.0).astype("float") * 1e-8
                + action_probability
            )
            return action, action_probability, log_action_probability

        return action_probability

    def get_config_dict(self, name: str = "") -> OmegaConf:
        config = OmegaConf.create({})
        if name != "":
            name += "/"
        config[f"{name}hidden_dim"] = self.hidden_dim
        config[f"{name}output_dim"] = self.output_dim

        if name == "":
            return config
        return {name: config}

    def get_search_space(self, prefix: str = ""):
        if prefix != "":
            prefix += "/"
        config = {f"{prefix}hidden_dim": tune.grid_search([32, 64, 128, 256])}
        return config


class Critic(Policy):
    """Critic for an Actor-Critic based algorithm"""

    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(nn.Dense(self.hidden_dim, name=f"linear1_{self.__repr__()}")(x))
        x = nn.relu(nn.Dense(self.hidden_dim, name=f"linear2_{self.__repr__()}")(x))
        return nn.Dense(self.output_dim, name=f"linear3_{self.__repr__()}")(x)

    def get_action(self, variables: VariableDict, state: jnp.ndarray) -> jnp.ndarray:
        return self.apply(variables, state)

    def get_config_dict(self, name: str = "") -> OmegaConf:
        config = OmegaConf.create({})
        if name != "":
            name += "/"
        config[f"{name}hidden_dim"] = self.hidden_dim
        config[f"{name}output_dim"] = self.output_dim

        if name == "":
            return config
        return {name: config}

    def get_search_space(self, prefix: str = ""):
        if prefix != "":
            prefix += "/"
        config = {f"{prefix}hidden_dim": tune.grid_search([32, 64, 128, 256])}
        return config
