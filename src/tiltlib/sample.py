from __future__ import annotations

from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from orix.crystal_map import CrystalMap, Phase
from orix.plot import IPFColorKeyTSL
from orix.plot.inverse_pole_figure_plot import _get_ipf_axes_labels
from orix.quaternion import Orientation, Rotation
from orix.vector import Vector3d, Miller
from orix.vector.fundamental_sector import _closed_edges_in_hemisphere
from orix.projections import StereographicProjection
from scipy.optimize import minimize
from hyperspy.signals import Signal1D
from hyperspy.roi import BaseROI, CircleROI, RectangularROI
from numba import njit

from tiltlib.sample_holder import Axis, SampleHolder


class Sample(SampleHolder):

    def __init__(self, oris: Orientation, phase: Phase, axes: list[Axis]) -> None:
        """Sample of spatially distinct orientations

        :param oris: The orientations at each point of the sample
        :type oris: Orientation
        :param phase: Crystallographic phase of the sample. Only single-phase crystals are supported
        :type phase: Phase
        :param axes: Tilt axes
        :type axes: list[Axis]
        """
        SampleHolder.__init__(self, axes)
        self.phase = phase
        self._original_rotations = Rotation(oris.data.copy())
        self.optical_axis_miller = Miller(uvw=[0, 0, 1], phase=self.phase)
        self.optical_axis = self.optical_axis_miller.unit

    @classmethod
    def from_crystal_map(cls, xmap: CrystalMap, axes: list[Axis]) -> "Sample":
        """Initialize a Sample from a CrystalMap

        :param xmap: Orientation map
        :type xmap: CrystalMap
        :param axes: Tilt axes
        :type axes: list[Axis]
        """
        oris = xmap.orientations.reshape(*xmap.shape)
        return cls(oris, xmap.phases[0], axes)

    @property
    def orientations(self) -> Orientation:
        """The orientations of the sample at the current tilts

        :return: Tilted orientations
        :rtype: Orientation
        """
        r = self._original_rotations
        r *= ~self._rotation
        o = Orientation(r.data, symmetry=self.phase.point_group)
        return o

    def crop(self, roi: BaseROI) -> "Sample":
        """Crop the sample with a hyperspy ROI and return a new cropped sample

        :param roi: Region of interest to keep
        :type roi: BaseROI
        :raises NotImplementedError: If an unsupported ROI is supplied
        :return: Cropped sample
        :rtype: Sample
        """
        if isinstance(roi, RectangularROI):
            top = int(roi.top)
            bottom = int(roi.bottom)
            left = int(roi.left)
            right = int(roi.right)
            slices = (slice(top, bottom, None), slice(left, right, None))
        elif isinstance(roi, CircleROI):
            cx = int(roi.cx)
            cy = int(roi.cy)
            r = roi.r
            x = np.arange(self._original_rotations.shape[1], dtype=float)
            y = np.arange(self._original_rotations.shape[0], dtype=float)
            x, y = np.meshgrid(x, y)
            x -= cx
            y -= cy
            mask = (x**2 + y**2) < r**2
            slices = mask
        else:
            raise NotImplementedError("Supported ROIs are RectangularROI and CircleROI")
        oris = self._original_rotations[slices]
        axes = self.axes
        return self.__class__(oris, self.phase, axes)

    def find_tilt_angles(
        self,
        zone_axis: Miller,
        degrees: bool = True,
        use_mean_orientation: bool = False,
    ) -> tuple[float, ...]:
        """Calculate the tilt angle(s) necessary to align the sample with a given optical axis

        :param zone_axis: The zone axis to align to
        :type zone_axis: Miller
        :param degrees: Whether to return degrees(True) or radians(False), defaults to True
        :type degrees: bool, optional
        :param use_mean_orientation: Whether to perform optimization using the mean orientation 
        of the sample(True) or the mean angle with the zone axis(False), defaults to False
        :type use_mean_orientation: bool, optional
        :return: Tilt angles
        :rtype: tuple[float, ...]
        """

        if use_mean_orientation:
            optimize = self._optimize_mean_orientation_func(zone_axis, degrees)
        else:
            optimize = self._optimize_angle_with_func(zone_axis, degrees)

        bounds = [(ax.min, ax.max) for ax in self.axes]
        angles = self.angles
        if degrees:
            bounds = np.rad2deg(bounds)
            angles = np.rad2deg(angles)

        res = minimize(
            optimize,
            angles,
            bounds=bounds,
            method="Nelder-Mead",
        )

        self.reset_rotation()

        return res.x

    def plot(self) -> plt.Figure:
        """Plot IPF colormap of the orientations at the given tilt angle(s)

        :return: IPFs
        :rtype: plt.Figure
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

    def plot_orientations(self) -> plt.Figure:
        """IPF scatterplot of orientations

        :return: IPFs
        :rtype: plt.Figure
        """
        fig = self.orientations.scatter(
            projection="ipf",
            return_figure=True,
            direction=Vector3d(np.eye(3)),
        )
        return fig

    def plot_interactive(
        self,
    ) -> tuple[plt.Figure, Slider]:
        """Make a IPF colormap plot with a slider for the tilt angle of each tilt axis

        :return: Figure and sliders. Returning the slider avoids their functionality 
        being deleted when the function returns
        :rtype: tuple[plt.Figure, Slider]
        """

        fig = plt.figure(layout="constrained")
        spec = GridSpec(3, 6, fig, height_ratios=[7, 1, 1])
        x_ax = fig.add_subplot(spec[0, 0:2])
        y_ax = fig.add_subplot(spec[0, 2:4])
        z_ax = fig.add_subplot(spec[0, 4:6])

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

        def update(_):
            self.rotate_to(*[slider.val for slider in sliders], degrees=True)
            x_im.set_data(ipfkey_x.orientation2color(self.orientations))
            y_im.set_data(ipfkey_y.orientation2color(self.orientations))
            z_im.set_data(ipfkey_z.orientation2color(self.orientations))
            fig.canvas.draw_idle()

        for i, tilt_axis in enumerate(self.axes):
            slider_ax = fig.add_subplot(spec[1 + i, 1:5])

            tilt_slider = Slider(
                slider_ax,
                f"Tilt axis {i + 1} angle [deg]",
                valmin=np.rad2deg(tilt_axis.min),
                valmax=np.rad2deg(tilt_axis.max),
                valinit=np.rad2deg(tilt_axis.angle),
                valfmt="%.2f",
                facecolor="#cc7000",
            )
            tilt_slider.on_changed(update)
            sliders.append(tilt_slider)

        return fig, tuple(sliders)

    def plot_orientations_interactive(self) -> tuple[plt.Figure, Slider]:
        """Interactive scatterplot of the orientations, with sliders to control the tilt axes

        :return: Figure and sliders. Returning the slider avoids their functionality 
        being deleted when the function returns
        :rtype: tuple[plt.Figure, Slider]
        """
        fig = plt.figure(layout="constrained")
        spec = GridSpec(3, 6, fig, height_ratios=[7, 1, 1])
        x_ax = fig.add_subplot(spec[0, 0:2])
        y_ax = fig.add_subplot(spec[0, 2:4])
        z_ax = fig.add_subplot(spec[0, 4:6])

        x_ax.axis("off")
        y_ax.axis("off")
        z_ax.axis("off")

        x_ax.set_aspect("equal")
        y_ax.set_aspect("equal")
        z_ax.set_aspect("equal")

        x_ax.set_title("x")
        y_ax.set_title("y")
        z_ax.set_title("z")

        proj = StereographicProjection()

        sector = self.phase.point_group.fundamental_sector
        edges = _closed_edges_in_hemisphere(sector.edges, sector)
        x_ax.plot(*proj.vector2xy(edges), c="black")
        y_ax.plot(*proj.vector2xy(edges), c="black")
        z_ax.plot(*proj.vector2xy(edges), c="black")

        labels = _get_ipf_axes_labels(sector.vertices, symmetry=self.phase.point_group)
        for l, x, y in zip(labels, *proj.vector2xy(sector.vertices)):
            x_ax.text(x, y, l, ha="center")
            y_ax.text(x, y, l, ha="center")
            z_ax.text(x, y, l, ha="center")

        oris = self.orientations.flatten()
        x_oris = (oris * Vector3d.xvector()).in_fundamental_sector(oris.symmetry)
        y_oris = (oris * Vector3d.yvector()).in_fundamental_sector(oris.symmetry)
        z_oris = (oris * Vector3d.zvector()).in_fundamental_sector(oris.symmetry)

        x_scatter = x_ax.scatter(*proj.vector2xy(x_oris))
        y_scatter = y_ax.scatter(*proj.vector2xy(y_oris))
        z_scatter = z_ax.scatter(*proj.vector2xy(z_oris))

        sliders: list[Slider] = []

        def update(_):
            self.rotate_to(*[slider.val for slider in sliders], degrees=True)
            oris = self.orientations.flatten()
            x_oris = (oris * Vector3d.xvector()).in_fundamental_sector(oris.symmetry)
            y_oris = (oris * Vector3d.yvector()).in_fundamental_sector(oris.symmetry)
            z_oris = (oris * Vector3d.zvector()).in_fundamental_sector(oris.symmetry)
            x_scatter.set_offsets(np.array(proj.vector2xy(x_oris)).T)
            y_scatter.set_offsets(np.array(proj.vector2xy(y_oris)).T)
            z_scatter.set_offsets(np.array(proj.vector2xy(z_oris)).T)
            fig.canvas.draw_idle()

        for i, tilt_axis in enumerate(self.axes):
            slider_ax = fig.add_subplot(spec[1 + i, 1:5])

            tilt_slider = Slider(
                slider_ax,
                f"Tilt axis {i + 1} angle [deg]",
                valmin=np.rad2deg(tilt_axis.min),
                valmax=np.rad2deg(tilt_axis.max),
                valinit=np.rad2deg(tilt_axis.angle),
                valfmt="%.2f",
                facecolor="#cc7000",
            )
            tilt_slider.on_changed(update)
            sliders.append(tilt_slider)
        return fig, tuple(sliders)

    def plot_angle_with(
        self,
        zone_axis: Miller,
        resolution: float = 1.0,
        use_mean_orientation: bool = False,
    ) -> plt.Figure:
        """Make a plot of similarity score as function of tilt angle(s).

        :param zone_axis: Zone axis to calculate for
        :type zone_axis: Miller
        :param resolution: Angular resolution of the tilt axes, in degrees, defaults to 1.0
        :type resolution: float, optional
        :param use_mean_orientation: Whether to perform optimization using the mean orientation 
        of the sample(True) or the mean angle with the zone axis(False), defaults to False
        :type use_mean_orientation: bool, optional
        :raises NotImplementedError: If more than 2 tilt axes are present, as up to 2 are supported
        :return: Line plot or colormap, depending on the number of tilt angles
        :rtype: plt.Figure
        """

        if use_mean_orientation:
            score = self._optimize_mean_orientation_func(zone_axis, degrees=True)
        else:
            score = self._optimize_angle_with_func(zone_axis, degrees=True)

        if len(self.axes) == 1:
            angles = np.arange(
                np.rad2deg(self.axes[0].min), np.rad2deg(self.axes[0].max), resolution
            )
            scores = [score(angle) for angle in angles]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(angles, scores)
            ax.set_xlabel("Tilt angle [deg]")
            ax.set_ylabel(
                f"Mean angle with [{zone_axis.u} {zone_axis.v} {zone_axis.w}]"
            )

        elif len(self.axes) == 2:
            angles_1 = np.arange(
                np.rad2deg(self.axes[0].min), np.rad2deg(self.axes[0].max), resolution
            )
            angles_2 = np.arange(
                np.rad2deg(self.axes[1].min), np.rad2deg(self.axes[1].max), resolution
            )
            scores = [
                score((angle_1, angle_2))
                for angle_2 in angles_2
                for angle_1 in angles_1
            ]
            scores = np.array(scores).reshape((angles_2.size, angles_1.size))

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(
                scores,
                extent=[
                    angles_1[0],
                    angles_1[-1],
                    angles_2[0],
                    angles_2[-1],
                ],
            )
            ax.set_xlabel("1st tilt angle [deg]")
            ax.set_ylabel("2nd tilt angle")
            fig.colorbar(im)

        else:
            raise NotImplementedError(
                "Only 1 and 2 tilt axes are supported for this plot"
            )
        self.reset_rotation()
        return fig

    def to_navigator(self) -> Signal1D:
        """Create a IPF-z colormap as a hyperspy signal

        :return: IPF-z
        :rtype: Signal1D
        """
        ipfkey = IPFColorKeyTSL(
            self.orientations.symmetry, direction=Vector3d.zvector()
        )

        float_rgb = ipfkey.orientation2color(self.orientations)
        int_rgb = (float_rgb * 255).astype(np.uint8)

        s = Signal1D(int_rgb)
        s.change_dtype("rgb8")

        return s

    def to_signal(self) -> Signal1D:
        """Hyperspy signal containing the quaternion data of the orientations at each sample point

        :return: Quaternion data
        :rtype: Signal1D
        """
        return Signal1D(self.orientations.data)

    def angle_with(self, zone_axis: Miller, degrees: bool = True) -> np.ndarray:
        """Calculate the angle between the optical axis and the target zone axis, 
        for all pixels in the sample.

        :param zone_axis: Zone axis to calculate for
        :type zone_axis: Miller
        :param degrees: Whether to return output in degrees(True) or radians(False), defaults to True
        :type degrees: bool, optional
        :return: Array of angles at all points in the sample
        :rtype: np.ndarray
        """
        vecs = (self.orientations * self.optical_axis).in_fundamental_sector()
        angles = _jit_angle_with(vecs.data, zone_axis.data)
        if degrees:
            angles = np.rad2deg(angles)
        return angles

    def mean_zone_axis(self) -> Miller:
        """Calculate the mean orientation in the sample, and return the zone axis.
        This is mostly useful for single-grain or cropped samples.

        :return: Zone axis
        :rtype: Miller
        """
        return (self.orientations.mean() * self.optical_axis_miller).round()

    def _optimize_angle_with_func(
        self, zone_axis: Miller, degrees: bool
    ) -> Callable[[tuple[float, ...]], float]:
        def optimize(angles) -> float:
            self.rotate_to(angles, degrees=degrees)
            aw = self.angle_with(zone_axis)
            return np.mean(aw)

        return optimize

    def _optimize_mean_orientation_func(
        self, zone_axis: Miller, degrees: bool
    ) -> Callable[[tuple[float, ...]], float]:
        o = self._original_rotations.mean()

        def optimize(angles) -> float:
            self.rotate_to(angles, degrees=degrees)
            ro = o * ~self._rotation
            mean_zone = (ro * self.optical_axis_miller).in_fundamental_sector()
            return mean_zone.angle_with(zone_axis, degrees=degrees)

        return optimize


@njit
def _jit_angle_with(vecs, target):
    dot = np.sum(vecs * target, axis=-1)
    norm = np.sqrt(np.sum(np.square(target)))
    cosines = dot / norm
    angles = np.arccos(cosines)
    return angles
