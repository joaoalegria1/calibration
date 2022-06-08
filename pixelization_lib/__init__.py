"""
_init.py__ . Part of calib_lib pacakge.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
07/06/2022: file creation
"""

from ._ellipse import (
    onSegment,
    orientation,
    doIntersect,
    is_inside_polygon,
)

__all__ = [
    "onSegment",
    "orientation",
    "doIntersect",
    "is_inside_polygon",
]
__version__ = "0.0.1"