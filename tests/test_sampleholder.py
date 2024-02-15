from __future__ import annotations

import numpy as np
import pytest

from .conftest import vec_eq, x, y, z


def test_import():
    from tiltlib.sample_holder import SampleHolder


def test_init(x, y, z):
    from tiltlib.axis import Axis
    from tiltlib.sample_holder import SampleHolder

    ax1 = Axis(x, 0, 0)
    b = SampleHolder((ax1,))
    assert isinstance(b, SampleHolder)

    ax1 = Axis(x, 0, 0)
    ax2 = Axis(y, 0, 0)
    b = SampleHolder((ax1,))

    assert ax1 in b.axes
    assert ax2 not in b.axes


def test_rotate_api(x, y, z):
    from tiltlib.sample_holder import Axis, SampleHolder

    ax1 = Axis(x, -50, 50, intrinsic=False)
    ax2 = Axis(y, -50, 50, intrinsic=True)
    b = SampleHolder((ax1, ax2))

    # too few angles
    with pytest.raises(ValueError):
        b.rotate_to(0)

    # too many angles
    with pytest.raises(ValueError):
        b.rotate(1, 2, 3, degrees=True)

    prev = b._rotation
    b.rotate_to(0, 0)

    assert prev is not b._rotation
    assert prev == b._rotation

    b.rotate_to(45, 0, degrees=True)

    assert prev != b._rotation

    prev = b._rotation
    b.rotate(-45, 0, degrees=True)

    assert np.allclose(b.to_matrix(), np.eye(3))


def test_rotate_rotations(x, y, z):
    from tiltlib.sample_holder import Axis, SampleHolder

    ax1 = Axis(x, -90, 90, intrinsic=True)
    b = SampleHolder([ax1])

    b.rotate_to(90, degrees=True)

    assert vec_eq(b.TEM_frame_to_sample_frame(x), x)
    assert vec_eq(b.TEM_frame_to_sample_frame(y), -z)
    assert vec_eq(b.TEM_frame_to_sample_frame(z), y)

    ax2 = Axis(y, -90, 90, intrinsic=True)

    b = SampleHolder([ax1, ax2])

    b.rotate_to(90, 90, degrees=True)

    assert vec_eq(b.TEM_frame_to_sample_frame(x), z)
    assert vec_eq(b.TEM_frame_to_sample_frame(y), x)
    assert vec_eq(b.TEM_frame_to_sample_frame(z), y)


def test_compare_to_manual(x, y, z):
    from tiltlib.sample_holder import Axis, SampleHolder

    ax1 = Axis(x, -90, 90, intrinsic=True)
    ax2 = Axis(y, -90, 90, intrinsic=True)
    b = SampleHolder([ax1, ax2])

    b.rotate_to(90, 90, degrees=True)

    assert vec_eq(b.TEM_frame_to_sample_frame(x), z)
    assert vec_eq(b.TEM_frame_to_sample_frame(y), x)
    assert vec_eq(b.TEM_frame_to_sample_frame(z), y)

    b.rotate(-90, 0, degrees=True)
    # Note: first axis is always extrinsic
    assert vec_eq(b.TEM_frame_to_sample_frame(x), z)
    assert vec_eq(b.TEM_frame_to_sample_frame(y), y)
    assert vec_eq(b.TEM_frame_to_sample_frame(z), -x)


def test_compare_to_scipy_euler(x, y, z, n_tests: int = 100):
    from scipy.spatial.transform import Rotation

    from tiltlib.sample_holder import Axis, SampleHolder

    axes = {
        "x": Axis(x, -90, 90, intrinsic=False),
        "X": Axis(x, -90, 90, intrinsic=True),
        "y": Axis(y, -90, 90, intrinsic=False),
        "Y": Axis(y, -90, 90, intrinsic=True),
        "z": Axis(z, -90, 90, intrinsic=False),
        "Z": Axis(z, -90, 90, intrinsic=True),
    }

    rng = np.random.default_rng(0)

    for _ in range(n_tests):
        t1, t2, t3 = rng.random(3) * 2 - 1

        possible_axes = ["X", "Y", "Z"]

        # need different consecutive axes
        chosen_axes = [rng.choice(possible_axes)]
        while len(chosen_axes) < 3:
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

        assert np.allclose(r.apply([1, 0, 0]), sh.sample_frame_to_TEM_frame(x).data)

        # # extrinsic
        order = order.lower()
        r = Rotation.from_euler(order, (t1, t2, t3))
        sh = SampleHolder([axes[a] for a in order])
        sh.rotate(t1, t2, t3)
        assert np.allclose(r.as_matrix(), sh.as_matrix())


def test_compare_to_orix(x, y, z):
    from orix.quaternion import Rotation
    from tiltlib.sample_holder import Axis, SampleHolder

    a1 = Rotation.from_axes_angles(-z, 90, degrees=True)

    b = SampleHolder([Axis(-z, -90, 90)])
    b.rotate(90, degrees=True)

    assert np.allclose(b.to_matrix(), a1.to_matrix())

    a2 = Rotation.from_axes_angles(x, 90, degrees=True)
    a = a2 * a1

    b = SampleHolder([b.axes[0], Axis(x, -90, 90, intrinsic=False)])
    b.rotate_to(90, 90, degrees=True)

    vax = a * x
    vay = a * y
    vaz = a * z
    vbx = b.sample_frame_to_TEM_frame(x)
    vby = b.sample_frame_to_TEM_frame(y)
    vbz = b.sample_frame_to_TEM_frame(z)
    assert vec_eq(vax, vbx)
    assert vec_eq(vay, vby)
    assert vec_eq(vaz, vbz)


def test_reset_rotation(x, y, z):
    from tiltlib.sample_holder import Axis, SampleHolder

    ax1 = Axis(x, -90, 90)
    ax2 = Axis(y, -90, 90)
    b = SampleHolder([ax1, ax2])

    b.rotate_to(10, 20, degrees=True)
    assert not np.allclose(b.rotation_matrix(), np.eye(3))

    b.reset_rotation()
    assert np.allclose(b.rotation_matrix(), np.eye(3))
