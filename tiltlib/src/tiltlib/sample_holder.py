from __future__ import annotations
from typing import Union, Iterable
import numpy as np
from orix.quaternion import Rotation
from orix.vector import Vector3d
from dataclasses import dataclass
import copy


@dataclass
class Axis:
    direction: Vector3d
    min: float
    max: float
    angle: float = 0
    degrees: bool = True
    intrinsic: bool = False

    def __post_init__(self):
        if self.degrees:
            self.min = np.deg2rad(self.min)
            self.max = np.deg2rad(self.max)
            self.degrees = False

    @property
    def extrinsic(self) -> bool:
        return not self.intrinsic

    def copy(self):
        return copy.copy(self)


class SampleHolder:
    def __init__(self, axes: list[Axis] = None) -> None:
        if axes is None:
            axes = []

        self.axes: list[Axis] = []
        for axis in axes:
            self.add_rotation_axis(axis)
        self._rotation = Rotation.identity()

    @property
    def angles(self) -> list[float]:
        return [axis.angle for axis in self.axes]

    @angles.setter
    def angles(self, angles: list[float]):
        self._check_angles(*angles)
        for i, angle in enumerate(angles):
            self.axes[i].angle = angle

    def _check_angles(self, *angles: float) -> None:
        if len(angles) == 1 and isinstance(angles, (list, tuple, np.ndarray)):
            angles = [angles[0]]
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

    def reset_rotation(self):
        self.angles = [0 for _ in self.axes]
        self._rotation = Rotation.identity()

    def add_rotation_axis(self, axis: Axis) -> None:
        if not isinstance(axis, Axis):
            raise ValueError("`axis` parameter must be of type `Axis`")
        self.axes.append(axis.copy())

    def rotate_to(self, *angles: float, degrees: bool = False):
        if degrees:
            angles = np.deg2rad(angles)
        self._check_angles(*angles)
        self.reset_rotation()
        self.angles = angles
        self._update_rotation()

    def rotate(self, *angles: float, degrees: bool = False):
        if degrees:
            angles = np.deg2rad(angles)
        self.angles = angles
        self._update_rotation()

    def _update_rotation(self):
        R = self._rotation

        for axis in self.axes:
            print(axis.direction, axis.angle)
            r = Rotation.from_axes_angles(axis.direction, axis.angle)

            if axis.extrinsic:
                R = r * R
            elif axis.intrinsic:
                R = R * r
            else:
                print("How did we get here?")

        self._rotation = R

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


class DoubleTiltHolder(SampleHolder):
    def __init__(self) -> None:
        super().__init__()
        self.add_rotation_axis(Axis(Vector3d.yvector(), 0, 0))
        self.add_rotation_axis(Axis(Vector3d.xvector(), 0, 0))

        def _add_rotation_axis(*args) -> None:
            raise NotImplementedError("Axes already defined")

        self.add_rotation_axis = _add_rotation_axis

    def set_y_tilt_limits(self, min: float, max: float, degrees: bool = True):
        self.axes[0].min = min
        self.axes[0].max = max
        self.axes[0].degrees = degrees
        self.axes[0].__post_init__()

    def set_x_tilt_limits(self, min: float, max: float, degrees: bool = True):
        self.axes[1].min = min
        self.axes[1].max = max
        self.axes[1].degrees = degrees
        self.axes[1].__post_init__()


class RotationHolder(SampleHolder):
    def __init__(self) -> None:
        super().__init__()
        self.add_rotation_axis(Axis(Vector3d.zvector(), 0, 0))
        self.add_rotation_axis(Axis(Vector3d.xvector(), 0, 0))

        def _add_rotation_axis(*args) -> None:
            raise NotImplementedError("Axes already defined")

        self.add_rotation_axis = _add_rotation_axis
