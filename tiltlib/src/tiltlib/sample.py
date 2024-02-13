from orix.crystal_map import CrystalMap
from tiltlib.sample_holder import Axis, SampleHolder
from orix.quaternion import Orientation, Rotation
import numpy as np


class Sample(SampleHolder):
    
    def __init__(self, xmap: CrystalMap, axes: list[Axis] = None) -> None:
        super().__init__(axes)
        self.xmap = xmap.deepcopy()
        self._original_xmap = xmap.deepcopy()

    def _update_xmap(self):
        self.xmap._rotations[...] = ~self._rotation * self._original_xmap._rotations

    def rotate_to(self, *angles: float, degrees: bool = False):
        super().rotate_to(*angles, degrees=degrees)
        self._update_xmap()

    def reset_rotation(self):
        super().reset_rotation()
        self.xmap = self._original_xmap.deepcopy()
    
    @property
    def orientations(self) -> Orientation:
        return self.xmap.orientations.reshape(*self.xmap.shape)

