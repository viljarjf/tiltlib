from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from orix.crystal_map import CrystalMap
from orix.plot import IPFColorKeyTSL
from orix.quaternion import Orientation, Rotation
from orix.vector import Vector3d

from tiltlib.sample_holder import Axis, SampleHolder


class Sample(SampleHolder):
    def __init__(self, xmap: CrystalMap, axes: list[Axis] = None) -> None:
        SampleHolder.__init__(self, axes)
        self.xmap = xmap.deepcopy()
        self._original_rotations = Rotation(xmap._rotations.data.copy())

    @classmethod
    def from_sampleholder(
        cls, xmap: CrystalMap, sampleholder: SampleHolder
    ) -> "Sample":
        return cls(xmap, sampleholder.axes)

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
        """Plot IPFs of the orientations at the given tilt angle(s)

        Returns:
            plt.Figure:
        """
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

        fig.tight_layout()
        return fig

    def plot_interactive(self) -> tuple[plt.Figure, Slider]:
        """Make a IPF plot with a slider for the tilt angle of each tilt axis

        Returns:
            tuple[plt.Figure, tuple[Slider, ...]]: The figure, and a tuple of all the sliders

        Note:
            The sliders are returned with the figure to avoid their functionality being deleted when the function returns
        """

        fig, ax = plt.subplots(1, 3, sharex="all", sharey="all")
        ax: tuple[plt.Axes, ...]
        (x_ax, y_ax, z_ax) = ax

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

        sliders: list[Slider] = []

        def update(_: float):
            self.rotate_to(*[slider.val for slider in sliders], degrees=True)
            x_im.set_data(ipfkey_x.orientation2color(self.orientations))
            y_im.set_data(ipfkey_y.orientation2color(self.orientations))
            z_im.set_data(ipfkey_z.orientation2color(self.orientations))
            fig.canvas.draw_idle()

        for i, tilt_axis in enumerate(self.axes):
            slider_ax = fig.add_axes([0.3, 0.05 + 0.1 * i, 0.4, 0.05])

            tilt_slider = Slider(
                slider_ax,
                "Tilt angle [deg]",
                valmin=np.rad2deg(tilt_axis.min),
                valmax=np.rad2deg(tilt_axis.max),
                valinit=np.rad2deg(tilt_axis.angle),
                valfmt="%.2f",
                facecolor="#cc7000",
            )
            tilt_slider.on_changed(update)
            sliders.append(tilt_slider)

        fig.tight_layout()

        return fig, tuple(sliders)
