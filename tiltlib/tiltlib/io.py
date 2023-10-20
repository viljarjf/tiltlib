"""Input/Output routines
"""
from __future__ import annotations
from orix.quaternion import Orientation, Rotation
import numpy as np
import yaml


def MTEX_2_Orix(mtex_ori: Orientation) -> Orientation:
    """Convert MTEX-orientations to orix coordinate definition

    :param mtex_ori: MTEX orientation
    :type mtex_ori: Orientation
    :return: The same orientation, in correct coordinate system
    :rtype: Orientation
    """
    rot_axes_align = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(30))

    return Orientation(rot_axes_align * (~mtex_ori), symmetry=mtex_ori.symmetry)


def get_ori_dict(fname: str) -> dict[str, list[float]]:
    """Get grain orientation data from file

    :param fname: Filepath
    :type fname: str
    :return: Grain orientation data
    :rtype: dict[str, list[float]]
    """
    with open(fname) as f:
        ori_dict = yaml.full_load(f)
    return ori_dict
