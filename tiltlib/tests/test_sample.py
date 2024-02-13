from tiltlib import Sample, Axis
from utils import x, get_xmap
import pytest

@pytest.mark.parametrize(
        "initial_angle",
        [0, -10, 10],
        )
def test_reset_rotation(initial_angle: float):
    original_xmap = get_xmap()
    s = Sample(get_xmap(), [Axis(x, -100, 100, initial_angle,  degrees=True)])

    assert s.xmap.orientations == original_xmap.orientations

    s.rotate(50, degrees=True)

    assert s.xmap.orientations != original_xmap.orientations

    s.reset_rotation()

    assert s.xmap.orientations == original_xmap.orientations

@pytest.mark.parametrize(
        ["angle_1", "angle_2"],
        [
            [0, 0],
            [0, 10],
            [-10, 10],
            [4, 20],
        ]
        )
def test_initial_angle(angle_1: float, angle_2: float):
    s1 = Sample(get_xmap(), [Axis(x, -100, 100, angle_1,  degrees=True)])
    s2 = Sample(get_xmap(), [Axis(x, -100, 100, angle_2,  degrees=True)])

    assert s1.orientations == s2.orientations

    s1.rotate(5, degrees=True)
    s2.rotate(5, degrees=True)

    assert s1.orientations == s2.orientations

    s1.rotate(-10, degrees=True)

    assert s1.orientations != s2.orientations

    s1.reset_rotation()
    s2.reset_rotation()

    assert s1.orientations == s2.orientations


