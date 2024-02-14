import copy
from dataclasses import dataclass

from numpy import deg2rad, rad2deg
from orix.quaternion import Rotation
from orix.vector import Vector3d


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
            self.min = deg2rad(self.min)
            self.max = deg2rad(self.max)
            self.angle = deg2rad(self.angle)
            self.degrees = False
        self._initial_angle = self.angle

    @property
    def R(self) -> Rotation:
        return Rotation.from_axes_angles(
            self.direction, self.angle - self._initial_angle
        )

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}:
        direction = [{self.direction.x[0]}, {self.direction.y[0]}, {self.direction.z[0]}]
        angle = {float(rad2deg(self.angle)) :.2f} degrees
        range: ({float(rad2deg(self.min)) :.2f}, {float(rad2deg(self.max)) :.2f}) degrees
        {'intrinsic' if self.intrinsic else 'extrinsic'}"""

    @property
    def extrinsic(self) -> bool:
        return not self.intrinsic

    @extrinsic.setter
    def extrinsic(self, b: bool):
        self.intrinsic = not b

    def copy(self):
        return copy.copy(self)
