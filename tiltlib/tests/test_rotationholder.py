from __future__ import annotations
from tqdm import tqdm
import numpy as np
import pytest
from orix.vector import Vector3d

def test_import():
    from tiltlib.sample_holder import SampleHolder

def test_init():
    from tiltlib.sample_holder import SampleHolder

    a = SampleHolder()
    from tiltlib.sample_holder import Axis

    ax1 = Axis(Vector3d.xvector(), 0, 0)
    b = SampleHolder((ax1,))

def test_add_axis():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(Vector3d.xvector(), 0, 0)
    ax2 = Axis(Vector3d.yvector(), 0, 0)
    b = SampleHolder((ax1,))

    assert ax1 in b.axes
    assert ax2 not in b.axes

    b.add_rotation_axis(ax2)

    assert ax1 in b.axes
    assert ax2 in b.axes

def test_rotate():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(Vector3d.xvector(), 0, 0, intrinsic=False)
    ax2 = Axis(Vector3d.yvector(), 0, 0, intrinsic=True)
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

    b.rotate_to(np.pi / 2, np.pi / 2)

    assert np.allclose(b.rotation_matrix(), [[0, 1, 0], [0, 0, -1], [-1, 0, 0]])

def test_intrinsic():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(Vector3d.xvector(), 0, 0, intrinsic=True)
    ax2 = Axis(Vector3d.yvector(), 0, 0, intrinsic=True)
    b = SampleHolder()
    b.add_rotation_axis(ax1)
    b.add_rotation_axis(ax2)

    b.rotate_to(90, 90, degrees=True)

    assert np.allclose(b.rotation_matrix(), [[0, 1, 0], [0, 0, -1], [-1, 0, 0]])

    b.rotate(-90, 90, degrees=True)

    assert np.allclose(b.to_matrix(), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

def test_extrinsic():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(Vector3d.xvector(), 0, 0, intrinsic=False)
    ax2 = Axis(Vector3d.yvector(), 0, 0, intrinsic=False)
    b = SampleHolder()
    b.add_rotation_axis(ax1)
    b.add_rotation_axis(ax2)

    b.rotate_to(-90, -90, degrees=True)

    assert np.allclose(b.rotation_matrix(), [[0, 0, -1], [1, 0, 0], [0, -1, 0]])

    b.rotate(-90, 0, degrees=True)

    assert np.allclose(b.to_matrix(), [[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    b.rotate(0, 90, degrees=True)

    assert np.allclose(b.to_matrix(), [[0, 1, 0], [0, 0, 1], [1, 0, 0]])

def test_compare_to_scipy(n_tests: int = 100):
    from scipy.spatial.transform import Rotation
    from tiltlib.sample_holder import SampleHolder, Axis
    import numpy as np

    axes = {
        "x": Axis(Vector3d.xvector(), 0, 0, intrinsic=False),
        "X": Axis(Vector3d.xvector(), 0, 0, intrinsic=True),
        "y": Axis(Vector3d.yvector(), 0, 0, intrinsic=False),
        "Y": Axis(Vector3d.yvector(), 0, 0, intrinsic=True),
        "z": Axis(Vector3d.zvector(), 0, 0, intrinsic=False),
        "Z": Axis(Vector3d.zvector(), 0, 0, intrinsic=True),
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
        sh.rotate(-t1, -t2, -t3)
        assert np.allclose(r.as_matrix(), sh.as_matrix())

        # extrinsic
        order = order.lower()
        r = Rotation.from_euler(order, (t1, t2, t3))
        sh = SampleHolder([axes[a] for a in order])
        sh.rotate(-t1, -t2, -t3)
        assert np.allclose(r.as_matrix(), sh.as_matrix())

def test():
    from orix.quaternion import Rotation
    from orix.vector import Vector3d
    r = Rotation.from_axes_angles((0, 0, -1), 90, degrees=True)
    print(r * Vector3d.xvector())
    print(r * Vector3d.yvector())
    print(r * Vector3d.zvector())


def test_reset_rotation():
    from tiltlib.sample_holder import SampleHolder, Axis

    ax1 = Axis(Vector3d.xvector(), 0, 0)
    ax2 = Axis(Vector3d.yvector(), 0, 0)
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
        ax1 = Axis(Vector3d.xvector(), 0, 0)
        a.add_rotation_axis(ax1)

def test_RotationHolder():
    from tiltlib.sample_holder import RotationHolder, Axis

    a = RotationHolder()

    a.rotate_to(90, 0, degrees=True)

    m = a.to_matrix()

    assert np.allclose(m, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    with pytest.raises(NotImplementedError):
        ax1 = Axis(Vector3d.xvector(), 0, 0)
        a.add_rotation_axis(ax1)
