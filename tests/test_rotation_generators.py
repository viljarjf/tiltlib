from __future__ import annotations

import numpy as np
from tqdm import tqdm


def test_rotation_generators(n_tests: int = 100, verbose=False):
    from .rotation_generator import GonioPosition, RotationGenerator

    rotgens = RotationGenerator.get_subclasses()
    test_range = range(n_tests)
    if verbose:
        test_range = tqdm(test_range)
    for _ in test_range:
        # Generate four random angles [-30, 30] degrees
        random_angles = np.random.default_rng(0).random(4) * 60 - 30
        gonio_1 = GonioPosition(random_angles[0], random_angles[1])
        gonio_2 = GonioPosition(random_angles[2], random_angles[3])

        a = RotationGenerator(gonio_1, gonio_2)
        for rotgen in rotgens:
            b = rotgen(gonio_1, gonio_2)
            if np.allclose(a.T, b.T):
                continue
            print(a.T)
            print(b.T)
        a = RotationGenerator(gonio_2, gonio_1)
        for rotgen in rotgens:
            b = rotgen(gonio_2, gonio_1)
            assert np.allclose(a.T, b.T)
