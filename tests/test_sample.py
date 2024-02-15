import pytest
from .conftest import default_xmap, x

from tiltlib import Axis, Sample


@pytest.mark.parametrize(
    "initial_angle",
    [0, -10, 10],
)
def test_reset_rotation(initial_angle: float, x, default_xmap):
    original_xmap = default_xmap
    s = Sample(default_xmap, [Axis(x, -100, 100, initial_angle, degrees=True)])

    assert s.orientations == original_xmap.orientations.reshape(*original_xmap.shape)

    s.rotate(50, degrees=True)

    assert s.orientations != original_xmap.orientations.reshape(*original_xmap.shape)

    s.reset_rotation()

    assert s.orientations == original_xmap.orientations.reshape(*original_xmap.shape)


@pytest.mark.parametrize(
    ["angle_1", "angle_2"],
    [
        [0, 0],
        [0, 10],
        [-10, 10],
        [4, 20],
    ],
)
def test_initial_angle(angle_1: float, angle_2: float, default_xmap, x):
    s1 = Sample(default_xmap, [Axis(x, -100, 100, angle_1, degrees=True)])
    s2 = Sample(default_xmap, [Axis(x, -100, 100, angle_2, degrees=True)])

    assert s1.orientations == s2.orientations

    s1.rotate(5, degrees=True)
    s2.rotate(5, degrees=True)

    assert s1.orientations == s2.orientations

    s1.rotate(-10, degrees=True)

    assert s1.orientations != s2.orientations

    s1.reset_rotation()
    s2.reset_rotation()

    assert s1.orientations == s2.orientations

    s1.rotate_to(angle_1 - 10, degrees=True)
    s2.rotate_to(angle_2 - 10, degrees=True)

    assert s1.orientations == s2.orientations


def test_rotate_to(default_xmap, x):
    s1 = Sample(default_xmap, [Axis(x, -100, 100, degrees=True)])
    s2 = Sample(default_xmap, [Axis(x, -100, 100, degrees=True)])

    assert s1.orientations == s2.orientations

    s1.rotate(5, degrees=True)
    s2.rotate_to(5, degrees=True)

    assert s1.orientations == s2.orientations

    s1.rotate(5, degrees=True)
    s2.rotate_to(10, degrees=True)  # 5 + 5 = 10

    assert s1.orientations == s2.orientations

    s1.rotate_to(-20, degrees=True)
    s2.rotate_to(-20, degrees=True)

    assert s1.orientations == s2.orientations

    s1.reset_rotation()
    s2.reset_rotation()

    assert s1.orientations == s2.orientations


def test_rotate_to_with_initial_angle(x, default_xmap):
    s1 = Sample(default_xmap, [Axis(x, -100, 100, 10, degrees=True)])
    s2 = Sample(default_xmap, [Axis(x, -100, 100, degrees=True)])

    assert s1.orientations == s2.orientations

    s1.rotate(-5, degrees=True)
    s2.rotate_to(5, degrees=True)

    assert all([a1 == a2 for a1, a2 in zip(s1.angles, s2.angles)])

    s1.rotate(5, degrees=True)
    s2.rotate_to(10, degrees=True)

    assert all([a1 == a2 for a1, a2 in zip(s1.angles, s2.angles)])

    s1.rotate_to(-20, degrees=True)
    s2.rotate_to(-30, degrees=True)

    assert s1.orientations == s2.orientations

    s1.reset_rotation()
    s2.rotate_to(10, degrees=True)

    assert all([a1 == a2 for a1, a2 in zip(s1.angles, s2.angles)])

    s2.reset_rotation()

    assert s1.orientations == s2.orientations


def test_rotating_to_initial_angle(x, default_xmap):
    s1 = Sample(default_xmap, [Axis(x, -100, 100, degrees=True)])

    s1.rotate_to(5, degrees=True)

    s2 = Sample(s1.xmap, [Axis(x, -100, 100, 5, degrees=True)])

    assert s1.orientations == s2.orientations

    s1.reset_rotation()
    s2.rotate_to(0, degrees=True)

    assert s1.orientations == s2.orientations

    s1.rotate_to(10, degrees=True)
    s2.rotate_to(10, degrees=True)

    assert s1.orientations == s2.orientations
