from orix.crystal_map import CrystalMap
from tiltlib.sample_holder import Axis, SampleHolder
from orix.quaternion import Orientation, Rotation
import numpy as np
from orix.plot import IPFColorKeyTSL
from orix.vector import Vector3d
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


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
    
    def plot(self) -> plt.Figure:
        oris = self.orientations
        symmetry = oris.symmetry
        ipfkey_x = IPFColorKeyTSL(symmetry, direction=Vector3d.xvector())
        ipfkey_y = IPFColorKeyTSL(symmetry, direction=Vector3d.yvector())
        ipfkey_z = IPFColorKeyTSL(symmetry, direction=Vector3d.zvector())

        fig = plt.figure()

        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(ipfkey_x.orientation2color(oris))
        ax.set_title("IPF x")
        ax.axis("off")

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(ipfkey_y.orientation2color(oris))
        ax.set_title("IPF y")
        ax.axis("off")

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(ipfkey_z.orientation2color(oris))
        ax.set_title("IPF z")
        ax.axis("off")

        return fig
    
    def plot_interactive(self) -> tuple[plt.Figure, Slider]:
        """Return the slider too to keep it working"""

        fig, ax = plt.subplots(1, 3, sharex="all", sharey="all")
        ax: tuple[plt.Axes, ...]
        (x_ax, y_ax, z_ax) = ax

        slider_ax = fig.add_axes([0.3, 0.05, 0.4, 0.05])

        tilt_slider = Slider(
            slider_ax, 
            'Tilt angle [deg]', 
            valmin=np.rad2deg(self.axes[0].min), 
            valmax=np.rad2deg(self.axes[0].max), 
            valinit=np.rad2deg(self.axes[0].angle), 
            valfmt='%.2f', 
            facecolor='#cc7000'
            )

        oris = self.orientations
        symmetry = oris.symmetry
        ipfkey_x = IPFColorKeyTSL(symmetry, direction=Vector3d.xvector())
        ipfkey_y = IPFColorKeyTSL(symmetry, direction=Vector3d.yvector())
        ipfkey_z = IPFColorKeyTSL(symmetry, direction=Vector3d.zvector())

        x_im = x_ax.imshow(ipfkey_x.orientation2color(oris))
        x_ax.set_title("IPF x")
        x_ax.axis("off")

        y_im = y_ax.imshow(ipfkey_y.orientation2color(oris))
        y_ax.set_title("IPF y")
        y_ax.axis("off")

        z_im = z_ax.imshow(ipfkey_z.orientation2color(oris))
        z_ax.set_title("IPF z")
        z_ax.axis("off")

        def update(tilt_angle: float):
            self.rotate_to(tilt_angle, degrees=True)
            x_im.set_data(ipfkey_x.orientation2color(self.orientations))
            y_im.set_data(ipfkey_y.orientation2color(self.orientations))
            z_im.set_data(ipfkey_z.orientation2color(self.orientations))
            fig.canvas.draw_idle()

        tilt_slider.on_changed(update)

        fig.tight_layout()

        return fig, tilt_slider
