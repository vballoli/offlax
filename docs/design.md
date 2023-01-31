# Design of Offlax

Offlax is inspired by [CleanRL's](https://github.com/vwxyzjn/cleanrl/) design i.e. individual script like package that can directly run from the command line. In addition to this, Offlax uses [Hydra](https://hydra.cc) for config management, [Ray](https://ray.io) to enable scaling and hyperparameter tuning and [Wandb](https://wandb.ai) for experiment tracking. Furthermore, my blog post on [Offline Reinforcement Learning](https://vballoli.github.io/research-recap/posts/offline-rl/) explains each of these algorithms supported within this framework, along with code excerpts.

