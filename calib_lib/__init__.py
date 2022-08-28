"""
_init.py__ . Part of calib_lib pacakge.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
31/01/2022: file creation
"""

from ._variables_pixel import *

from ._calib import (
    get_calibration_matrix,
    get_rotation_matrix,
    get_translation_matrix,
    get_extrinsic_matrix,
    get_projection_matrix,
    get_plucker_matrix,
    get_image_points,
    calibration,
    create_1024_point_chess,
    create_1024_circle)

from ._DLT_lib import DLT, SVD
from ._homogeneous import to_homogeneus_arr, to_homogeneous, to_inhomogeneous
from ._checkerboard import (
    create_chessboard,
    get_chessboard_rot_trans,
    createcircle_3d,
    create_chessboard_4points,
    create_chessboard_1024points,
    get_plane)
from ._normalization import normalization,denormalization
from ._image import Image, ImagePlane
from ._figures import GenericPoint, Polygon
from ._frame import ReferenceFrame
from ._principal_axis import PrincipalAxis

from ._utils import (
    draw3d_arrow,
    get_plane_from_three_points,
    set_xyzlim3d,
    set_xyzticks,
)

__all__ = [
    "SVD",
    "DLT",
    "GenericPoint",
    "Image",
    "ImagePlane",
    "Polygon",
    "PrincipalAxis",
    "ReferenceFrame",
    "draw3d_arrow",
    "get_calibration_matrix",
    "get_plane_from_three_points",
    "get_plucker_matrix",
    "get_projection_matrix",
    "get_rotation_matrix",
    "to_homogeneous",
    "to_homogeneus_arr",
    "to_inhomogeneous",
    "set_xyzlim3d",
    "set_xyzticks",
    "create_chessboard",
    "get_chessboard_rot_trans",
    "get_plane",
    "normalization",
    "denormalization",
    "get_translation_matrix",
    "get_extrinsic_matrix",
    "get_rotation_matrix",
    "get_image_points",
    "calibration",
    "createcircle_3d",
    "create_chessboard_4points",
    "create_chessboard_1024points",
    "create_1024_point_chess",
    "create_1024_circle"
]
__version__ = "0.0.2"