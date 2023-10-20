from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as SpRotation
from orix.quaternion import Rotation
from orix.vector import Vector3d

from tiltlib.sample_holder import SampleHolder, Axis


class GonioPosition:
    def __init__(self, x_tilt, y_tilt, degrees=True) -> None:
        if degrees:
            self.x_tilt = np.radians(x_tilt)
            self.y_tilt = np.radians(y_tilt)
        else:
            self.x_tilt = x_tilt
            self.y_tilt = y_tilt

    def __repr__(self) -> str:
        return f"x_tilt: {np.rad2deg(self.x_tilt) :.2f}, y_tilt: {np.rad2deg(self.y_tilt) :.2f}"


class RotationGenerator:
    def __init__(
        self,
        new_gonjo_pos: GonioPosition,
        old_gonjo_pos: GonioPosition = GonioPosition(0, 0),
    ) -> None:
        self.new_gonjo_pos = new_gonjo_pos
        self.old_gonjo_pos = old_gonjo_pos

        self.alpha_0 = self.old_gonjo_pos.x_tilt
        self.beta_0 = self.old_gonjo_pos.y_tilt

        self.alpha = self.new_gonjo_pos.x_tilt - self.old_gonjo_pos.x_tilt
        self.beta = self.new_gonjo_pos.y_tilt

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}: 
        alpha:   {self.alpha :.2f}
        alpha_0: {self.alpha_0 :.2f}
        beta:    {self.beta :.2f}
        beta_0:  {self.beta_0 :.2f}
        """

    @property
    def T1(self):
        a11 = 1
        a12 = 0
        a13 = 0
        a21 = 0
        a22 = np.cos(self.alpha)
        a23 = -np.sin(self.alpha)
        a31 = 0
        a32 = np.sin(self.alpha)
        a33 = np.cos(self.alpha)

        return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    @property
    def T2(self):
        a11 = np.cos(self.beta)
        a12 = -np.sin(self.beta) * np.sin(self.alpha_0)
        a13 = np.sin(self.beta) * np.cos(self.alpha_0)
        a21 = np.sin(self.beta) * np.sin(self.alpha_0)
        a22 = np.cos(self.alpha_0) ** 2 + np.sin(self.alpha_0) ** 2 * np.cos(self.beta)
        a23 = np.sin(self.alpha_0) * np.cos(self.alpha_0) * (1 - np.cos(self.beta))
        a31 = -np.sin(self.beta) * np.cos(self.alpha_0)
        a32 = np.sin(self.alpha_0) * np.cos(self.alpha_0) * (1 - np.cos(self.beta))
        a33 = np.sin(self.alpha_0) ** 2 + np.cos(self.alpha_0) ** 2 * np.cos(self.beta)

        return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    @property
    def T(self):
        return self.T1 @ self.T2

    # Register subclasses
    __subclasses = []

    def __init_subclass__(cls) -> None:
        cls.__subclasses.append(cls)

    @staticmethod
    def get_subclasses() -> list[type[RotationGenerator]]:
        return RotationGenerator.__subclasses


class ScipyRotationGenerator(RotationGenerator):
    @property
    def T1(self) -> np.ndarray:
        return SpRotation.from_euler("X", self.alpha).as_matrix()

    @property
    def T2(self) -> np.ndarray:
        X1 = SpRotation.from_euler("X", -self.alpha_0).as_matrix()
        Y = SpRotation.from_euler("Y", self.beta).as_matrix()
        X2 = SpRotation.from_euler("X", self.alpha_0).as_matrix()
        return X2 @ Y @ X1

    @property
    def _T(self) -> np.ndarray:
        return SpRotation.from_euler(
            "xyx",
            [
                -self.old_gonjo_pos.x_tilt,
                self.new_gonjo_pos.y_tilt,
                self.new_gonjo_pos.x_tilt,
            ],
        ).as_matrix()


class OrixRotationGenerator(RotationGenerator):
    @property
    def _T1(self) -> Rotation:
        T1_axis = Vector3d.xvector()
        return Rotation.from_axes_angles(T1_axis, self.alpha)

    @property
    def _T2(self) -> Rotation:
        T2_axis = Vector3d.yvector().rotate(Vector3d.xvector(), self.alpha_0)
        return Rotation.from_axes_angles(T2_axis, self.beta)

    @property
    def T1(self) -> np.ndarray:
        return self._T1.to_matrix()

    @property
    def T2(self) -> np.ndarray:
        return self._T2.to_matrix()


class SampleHolderRotationGenerator(RotationGenerator):
    def __init__(
        self,
        new_gonjo_pos: GonioPosition,
        old_gonjo_pos: GonioPosition = GonioPosition(0, 0),
    ) -> None:
        super().__init__(new_gonjo_pos, old_gonjo_pos)

        self.sampleholder = SampleHolder()

        T2 = Vector3d.yvector().rotate(Vector3d.xvector(), self.alpha_0)
        self.sampleholder.add_rotation_axis(Axis(T2, 0, 0, intrinsic=False))

        T1 = Vector3d.xvector()
        self.sampleholder.add_rotation_axis(Axis(T1, 0, 0, intrinsic=True))

        self.sampleholder.rotate(self.beta, self.alpha)

    @property
    def T(self):
        return self.sampleholder.to_matrix()
