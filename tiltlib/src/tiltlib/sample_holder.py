from __future__ import annotations
import numpy as np
from orix.vector import Vector3d

from tiltlib.axis import Axis

class SampleHolder:
    def __init__(self, axes: list[Axis] = None) -> None:
        """The first axis is always extrinsic.

        Args:
            axes (list[Axis], optional): axes, in order
        """
        if axes is None:
            axes = []
        if isinstance(axes, Axis):
            axes = [axes]

        if not all(isinstance(axis, Axis) for axis in axes):
            raise ValueError("`axes` iterable can only contain `Axis` objects")

        self.axes = [axis.copy() for axis in axes]
        self.axes[0].extrinsic = True
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:\n" + '\n'.join(str(ax) for ax in self.axes)

    @property
    def angles(self) -> list[float]:
        return list(axis.angle for axis in self.axes)

    @property
    def _initial_angles(self) -> list[float]:
        return list(axis._initial_angle for axis in self.axes)

    @angles.setter
    def angles(self, angles: list[float]):
        self._check_angles(angles)
        for i, angle in enumerate(angles):
            self.axes[i].angle = float(angle)

    def _check_angles(self, angles: list[float]) -> None:
        if len(angles) == 1 and isinstance(angles[0], (list, tuple, np.ndarray)):
            angles = angles[0]
        if len(angles) > len(self.angles):
            raise ValueError(
                f"Too many angles. Expexted {len(self.angles)}, got {len(angles)}"
            )
        if len(angles) < len(self.angles):
            raise ValueError(
                f"Too few angles. Expexted {len(self.angles)}, got {len(angles)}"
            )
        if not all(isinstance(angle, (float, int)) for angle in angles):
            raise ValueError("All angles must be numeric") 
        if not all(ax.min <= angle <= ax.max for angle, ax in zip(angles, self.axes)):
            raise IndexError("Angle out of range for one or more axes" + f"\n{angles = }\n{self.angles = }\n{self.axes = }")

    def reset_rotation(self):
        self.rotate_to(self._initial_angles)

    def rotate_to(self, *angles: float, degrees: bool = False):
        if len(angles) == 1 and isinstance(angles[0], (list, tuple, np.ndarray)):
            angles = angles[0]
        if degrees:
            angles = np.deg2rad(angles)
        self.angles = angles

    def rotate(self, *angles: float, degrees: bool = False):
        if len(angles) == 1 and isinstance(angles[0], (list, tuple, np.ndarray)):
            angles = angles[0]
        if degrees:
            angles = list(np.deg2rad(angles))
        self._check_angles(angles)
        angles = [current + target for target, current in zip(angles, self.angles)]
        self.rotate_to(*angles)

    @property
    def _rotation(self):
        R = self.axes[0].R

        for axis in self.axes[1:]:
            if axis.intrinsic:
                R = R * axis.R
            else:
                R = axis.R * R

        return R

    def rotation_matrix(self) -> np.ndarray:
        return self._rotation.to_matrix().squeeze()

    def to_matrix(self) -> np.ndarray:
        return self.rotation_matrix()

    def as_matrix(self) -> np.ndarray:
        return self.rotation_matrix()

    def sample_frame_to_TEM_frame(self, v: Vector3d) -> Vector3d:
        return self._rotation * v

    def TEM_frame_to_sample_frame(self, v: Vector3d) -> Vector3d:
        return ~self._rotation * v
