import os

import numpy as np
from petastorm import make_reader

from offlax.replay_buffer import ReplayBuffer


def test_replay_buffer():
    observations = np.random.randn(1000, 10)
    ReplayBuffer(
        observations,
        np.random.randn(1000, 10),
        np.random.randn(1000, 10),
        np.random.randn(1000, 10),
    ).dump("file:///tmp/test_offlax")
    assert os.path.isdir("/tmp/test_offlax")
