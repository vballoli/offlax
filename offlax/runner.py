from typing import Dict

import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback

__all__ = ["OfflaxRunner"]


# @ray.remote
class OfflaxRunner:
    """
    Generic runner that encapsulates search, training and inference.

    Args:
        algorithm: Offlax Algorithm to run
        search_space (_type_): Search space of the algorithm. Defaults to the algorithm's search space if unspecified
    """

    def __init__(
        self,
        algorithm: str,
        search_space=None,
        enable_wandb: bool = True,
        wandb_kwargs: Dict = {},
        *args,
        **kwargs
    ):
        ray.init(*args, **kwargs)
        self.algorithm = algorithm
        if search_space is None:
            self.search_space = self.algorithm.get_search_space()
        else:
            self.search_space = search_space
        self.enable_wandb = enable_wandb
        self.wandb_kwargs = wandb_kwargs

        self.callbacks = self._get_callbacks()

    def _get_callbacks(self):
        callbacks = None
        if self.enable_wandb:
            callbacks = [WandbLoggerCallback(**self.wandb_kwargs)]
        return callbacks

    def search(self, environment: str):
        tuner = tune.Tuner(
            tune.with_parameters(self.train, environment=environment),
            param_space=self.search_space,
            run_config=air.RunConfig(callbacks=self.callbacks),
        )
        results = tuner.fit()
        return results

    def train(self, config=None, environment: str = "maze2d-open-v0"):
        if config is not None:
            algorithm = self.algorithm.parse_config(config)
            algorithm.train(
                rtune=tune, enable_wandb=self.enable_wandb, environment=environment
            )
        else:
            algorithm = self.algorithm
            algorithm.train(enable_wandb=self.enable_wandb, environment=environment)

    def test(self):
        pass
