from typing import Callable

import gym
import numpy as np

from offlax.replay_buffer import ReplayBuffer


def generate_offlax_dataset(
    env: gym.Env, agent: Callable, steps: int, path: str
) -> None:
    """Generates an offlax dataset for the environment and agent

    Args:
        env (gym.Env): An Gym API compatible environment
        agent (Callable): a function that returns an action given the state of the environment
        steps (int): number of steps
    """
    step_count = 0
    obs = env.reset()

    obs_v = None
    action_v = None
    reward_v = None
    done_v = None

    while step_count < steps:
        action = agent(obs)
        next_obs, reward, done, info = env.step(action)

        if obs_v:
            obs_v = np.hstack([obs_v, action])
        else:
            obs_v = obs

        if action_v:
            action_v = np.hstack([action_v, action])
        else:
            action_v = action

        if reward_v:
            reward_v = np.hstack([reward_v, action])
        else:
            reward_v = reward

        if done_v:
            done_v = np.hstack([done_v, action])
        else:
            done_v = done

        obs = next_obs

    ReplayBuffer(obs_v, action_v, reward_v, done_v).dump(path)
