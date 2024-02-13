from orix.crystal_map import CrystalMap
from tiltlib.sample_holder import Axis, SampleHolder
from orix.quaternion import Orientation, Rotation
import numpy as np


class Sample(SampleHolder):
    
    def __init__(self, xmap: CrystalMap, axes: list[Axis] = None) -> None:
        super().__init__(axes)
        self.xmap = xmap
        self._xmap_rotations = Rotation(self.xmap._rotations.data.copy()) * self._rotation
        self._grain_masks = []

    def _update_xmap(self):
        self.xmap._rotations[...] = ~self._rotation * self._xmap_rotations

    def rotate_to(self, *angles: float, degrees: bool = False):
        super().rotate_to(*angles, degrees=degrees)
        self._update_xmap()

    def add_grain(self, grain_mask: np.ndarray):
        self._grain_masks.append(grain_mask)
    
    @property
    def orientations(self) -> Orientation:
        return self.xmap.orientations.reshape(*self.xmap.shape)

