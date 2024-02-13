from orix.crystal_map import CrystalMap
from tiltlib.sample_holder import Axis, SampleHolder
from orix.quaternion import Orientation, Rotation
import numpy as np


class Sample(SampleHolder):
    
    def __init__(self, xmap: CrystalMap, axes: list[Axis] = None) -> None:
        SampleHolder.__init__(self, axes)
        self.xmap = xmap.deepcopy()
        self._original_rotations = Rotation(xmap._rotations.data.copy())

    def _update_xmap(self):
        self.xmap._rotations[...] = self._original_rotations * ~self._rotation

    def rotate_to(self, *angles: float, degrees: bool = False):
        SampleHolder.rotate_to(self, *angles, degrees=degrees)
        self._update_xmap()

    # TODO fix coordinates changing in the xmap maybe
    
    @property
    def orientations(self) -> Orientation:
        return self.xmap.orientations.reshape(*self.xmap.shape)

