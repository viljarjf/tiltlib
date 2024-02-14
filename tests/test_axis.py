from __future__ import annotations

import numpy as np
from orix.vector import Vector3d


def test_import():
    from tiltlib.sample_holder import Axis


def test_init():
    from tiltlib.sample_holder import Axis

    a = Axis(Vector3d.xvector(), 0, 0)
    b = Axis(Vector3d(np.random.random(3)), 0, 0)


def test_degree():
    from tiltlib.sample_holder import Axis

    mi, ma = -10, 10
    v = Axis(Vector3d(np.random.random(3)), mi, ma, degrees=False)
    assert mi == v.min
    assert ma == v.max
    v = Axis(Vector3d(np.random.random(3)), mi, ma, degrees=True)
    assert mi != v.min
    assert ma != v.max
    assert np.isclose(np.deg2rad(mi), v.min)
    assert np.isclose(np.deg2rad(ma), v.max)


def test_copy():
    from tiltlib.sample_holder import Axis

    a = Axis(Vector3d.xvector(), 0, 0)
    b = a.copy()
    assert a is not b
    assert a == b

    b.direction = Vector3d.yvector()
    assert a.direction != b.direction
    assert a != b

    b = a.copy()
    b.direction *= 2
    assert a.direction != b.direction
    assert a != b
