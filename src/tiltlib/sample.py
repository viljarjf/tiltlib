from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from orix.crystal_map import CrystalMap
from orix.plot import IPFColorKeyTSL
from orix.quaternion import Orientation, Rotation
from orix.vector import Vector3d, Miller
from scipy.optimize import minimize
from hyperspy.signals import Signal1D
from hyperspy.roi import BaseROI, CircleROI, RectangularROI

from tiltlib.sample_holder import Axis, SampleHolder


class Sample(SampleHolder):
    def __init__(self, xmap: CrystalMap, axes: list[Axis] = None) -> None:
        SampleHolder.__init__(self, axes)
        self.xmap = xmap.deepcopy()
        self._original_rotations = Rotation(xmap._rotations.data.copy())
        self._slices = (slice(None, None, None), slice(None, None, None))

    @classmethod
    def from_sampleholder(
        cls, xmap: CrystalMap, sampleholder: SampleHolder
    ) -> "Sample":
        return cls(xmap, sampleholder.axes)

    def _update_xmap(self):
        self.xmap._rotations[:, 0] = self._original_rotations[:, 0] * ~self._rotation

    def rotate_to(self, *angles: float, degrees: bool = False):
        SampleHolder.rotate_to(self, *angles, degrees=degrees)
        self._update_xmap()

    # TODO fix coordinates changing in the xmap maybe

    @property
    def orientations(self) -> Orientation:
        o = self.xmap.orientations.reshape(*self.xmap.shape)[self._slices]
        return o

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

    def crop(self, roi: BaseROI) -> "Sample":
        """Crop the sample with a hyperspy ROI and return a new cropped sample"""
        out = self.__class__(self.xmap, self.axes)
        if isinstance(roi, RectangularROI):
            top = int(roi.top)
            bottom = int(roi.bottom)
            left = int(roi.left)
            right = int(roi.right)
            out._slices = (slice(top, bottom, None), slice(left, right, None))
        elif isinstance(roi, CircleROI):
            cx = int(roi.cx)
            cy = int(roi.cy)
            r = roi.r
            x = np.arange(self.xmap.shape[1], dtype=float)
            y = np.arange(self.xmap.shape[0], dtype=float)
            x, y = np.meshgrid(x, y)
            x -= cx
            y -= cy
            mask = (x**2 + y**2) < r**2
            out._slices = mask
        else:
            raise NotImplementedError("Supported ROIs are RectangularROI and CircleROI")
        return out

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

    def find_tilt_angles(self, zone_axis: Miller) -> tuple[float, ...]:
        """Calculate the tilt angle(s) necessary to align the sample with a given optical axis

        Args:
            zone_axis (Miller): desired zone axis

        Returns:
            tuple[float, ...]: Tilt angles for each axis

        """

        optical_axis = Miller(uvw=[0, 0, 1], phase=self.xmap.phases[0])

        def angle_with(angles) -> np.ndarray:
            """Calculate the angle between the optical axis and the target zone axis for all pixels in the sample. Flattened output."""
            self.rotate_to(*angles, degrees=True)
            return (
                (self.orientations * optical_axis)
                .in_fundamental_sector()
                .angle_with(zone_axis, degrees=True)
            )

        def optimize(angles) -> float:
            """Mean of the bottom 2/3 of angles"""
            aw = angle_with(angles)
            k = aw.size // 3 * 2
            return np.mean(np.partition(aw, k, axis=None)[:k])

        res = minimize(
            optimize,
            self.angles,
            bounds=np.rad2deg([(ax.min, ax.max) for ax in self.axes]),
            method="Nelder-Mead",
        )

        self.reset_rotation()

        return res.x

    def to_navigator(self) -> Signal1D:
        """Get a IPF"""
        # Might not actually be 2D, but the navigation signal is...

        ipfkey = IPFColorKeyTSL(
            self.orientations.symmetry, direction=Vector3d.zvector()
        )

        float_rgb = ipfkey.orientation2color(self.orientations)

        int_rgb = (float_rgb * 255).astype(np.uint8)

        s = Signal1D(int_rgb)

        s.change_dtype("rgb8")

        return s

    def to_signal(self) -> Signal1D:
        return Signal1D(self.orientations.data)
