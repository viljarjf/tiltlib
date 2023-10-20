from __future__ import annotations
import numpy as np
import pytest
from orix.vector import Vector3d

from utils import vec_eq, x, y, z

def test_import():
    from tiltlib.sample_holder import SampleHolder

def test_init():
    from tiltlib.sample_holder import SampleHolder

    a = SampleHolder()
    from tiltlib.sample_holder import Axis

    ax1 = Axis(x, 0, 0)
    b = SampleHolder((ax1,))

def test_add_axis():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(x, 0, 0)
    ax2 = Axis(y, 0, 0)
    b = SampleHolder((ax1,))

    assert ax1 in b.axes
    assert ax2 not in b.axes

    b.add_rotation_axis(ax2)

    assert ax1 in b.axes
    assert ax2 in b.axes

def test_rotate_api():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(x, 0, 0, intrinsic=False)
    ax2 = Axis(y, 0, 0, intrinsic=True)
    b = SampleHolder()
    b.add_rotation_axis(ax1)
    b.add_rotation_axis(ax2)

    # too few angles
    with pytest.raises(ValueError):
        b.rotate_to(0)

    # too many angles
    with pytest.raises(ValueError):
        b.rotate(1, 2, 3)

    prev = b._rotation
    b.rotate_to(0, 0)

    assert prev is not b._rotation
    assert prev == b._rotation

    b.rotate_to(45, 0, degrees=True)

    assert prev != b._rotation

    prev = b._rotation
    b.rotate(-45, 0, degrees=True)

    assert np.allclose(b.to_matrix(), np.eye(3))

def test_rotate_rotations():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(x, 0, 0, intrinsic=True)
    b = SampleHolder([ax1])

    b.rotate_to(90, degrees=True)

    assert vec_eq(b.TEM_frame_to_sample_frame(x), x)
    assert vec_eq(b.TEM_frame_to_sample_frame(y), -z)
    assert vec_eq(b.TEM_frame_to_sample_frame(z), y)

    ax2 = Axis(y, 0, 0, intrinsic=True)

    b.add_rotation_axis(ax2)

    b.rotate_to(90, 90, degrees=True)

    assert vec_eq(b.TEM_frame_to_sample_frame(x), z)
    assert vec_eq(b.TEM_frame_to_sample_frame(y), x)
    assert vec_eq(b.TEM_frame_to_sample_frame(z), y)


def test_intrinsic():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(x, 0, 0, intrinsic=True)
    ax2 = Axis(y, 0, 0, intrinsic=True)
    b = SampleHolder()
    b.add_rotation_axis(ax1)
    b.add_rotation_axis(ax2)

    b.rotate_to(90, 90, degrees=True)

    assert vec_eq(b.TEM_frame_to_sample_frame(x), z)

    b.rotate(-90, 90, degrees=True)

    # assert np.allclose(b.to_matrix(), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

def test_extrinsic():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(x, 0, 0, intrinsic=False)
    ax2 = Axis(y, 0, 0, intrinsic=False)
    b = SampleHolder()
    b.add_rotation_axis(ax1)
    b.add_rotation_axis(ax2)

    b.rotate_to(-90, -90, degrees=True)

    assert vec_eq(b.TEM_frame_to_sample_frame(x), y)
    assert vec_eq(b.TEM_frame_to_sample_frame(y), z)
    assert vec_eq(b.TEM_frame_to_sample_frame(z), x)

    b.rotate(-90, 0, degrees=True)

    assert vec_eq(b.TEM_frame_to_sample_frame(x), y)
    assert vec_eq(b.TEM_frame_to_sample_frame(y), x)
    assert vec_eq(b.TEM_frame_to_sample_frame(z), -z)
    
    b.rotate(0, 90, degrees=True)

    assert vec_eq(b.TEM_frame_to_sample_frame(x), -z)
    assert vec_eq(b.TEM_frame_to_sample_frame(y), x)
    assert vec_eq(b.TEM_frame_to_sample_frame(z), -y)

def test_compare_to_scipy(n_tests: int = 100):
    from scipy.spatial.transform import Rotation
    from tiltlib.sample_holder import SampleHolder, Axis

    axes = {
        "x": Axis(x, 0, 0, intrinsic=True),
        "X": Axis(x, 0, 0, intrinsic=False),
        "y": Axis(y, 0, 0, intrinsic=True),
        "Y": Axis(y, 0, 0, intrinsic=False),
        "z": Axis(z, 0, 0, intrinsic=True),
        "Z": Axis(z, 0, 0, intrinsic=False),
    }

    rng = np.random.default_rng(0)
    
    for _ in range(n_tests):
        t1, t2, t3 = rng.random(3) * 2 - 1
        
        possible_axes = ["X", "Y", "Z"]

        # need different consecutive axes
        chosen_axes = [rng.choice(possible_axes)]
        for _ in range(2):
            next_axis = rng.choice(possible_axes)
            while next_axis == chosen_axes[-1]:
                next_axis = rng.choice(possible_axes)
            chosen_axes.append(next_axis)

        order = "".join(chosen_axes)
    
        # intrinsic
        r = Rotation.from_euler(order, (t1, t2, t3))
        sh = SampleHolder([axes[a] for a in order])
        sh.rotate(t1, t2, t3)
        assert np.allclose(r.as_matrix(), sh.as_matrix())

        # # extrinsic
        order = order.lower()
        r = Rotation.from_euler(order, (t1, t2, t3))
        sh = SampleHolder([axes[a] for a in order])
        sh.rotate(t1, t2, t3)
        assert np.allclose(r.as_matrix(), sh.as_matrix())

def test_compare_to_orix():
    from orix.quaternion import Rotation
    a = Rotation.from_axes_angles((0, 0, -1), 90, degrees=True)
    assert np.allclose(a.to_matrix(), [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    from tiltlib.sample_holder import SampleHolder, Axis
    b = SampleHolder([Axis(-z, 0, 0)])
    b.rotate(90, degrees=True)

    assert np.allclose(b.to_matrix(), [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    print("test")
    print(b.TEM_frame_to_sample_frame(x))


def test_reset_rotation():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(x, 0, 0)
    ax2 = Axis(y, 0, 0)
    b = SampleHolder()
    b.add_rotation_axis(ax1)
    b.add_rotation_axis(ax2)

    b.rotate_to(10, 20, degrees=True)
    assert not np.allclose(b.rotation_matrix(), np.eye(3))

    b.reset_rotation()
    assert np.allclose(b.rotation_matrix(), np.eye(3))

def test_DoubleTiltHolder():
    from tiltlib.sample_holder import DoubleTiltHolder, Axis

    a = DoubleTiltHolder()

    a.rotate_to(90, 0, degrees=True)

    m = a.to_matrix()

    assert np.allclose(m, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    with pytest.raises(NotImplementedError):
        ax1 = Axis(x, 0, 0)
        a.add_rotation_axis(ax1)

def test_RotationHolder():
    from tiltlib.sample_holder import RotationHolder, Axis

    a = RotationHolder()

    a.rotate_to(90, 0, degrees=True)

    m = a.to_matrix()

    assert np.allclose(m, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    with pytest.raises(NotImplementedError):
        ax1 = Axis(x, 0, 0)
        a.add_rotation_axis(ax1)
