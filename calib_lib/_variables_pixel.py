"""
variables file. Part of calib_lib package.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
14/06/2022: file creation
-----------------------
"""

import numpy as np
from calib_lib import *

F = 16                                       # focal length( in mm )
gridSize = 1024
subpixel_grid = 64
image_size = np.array([1936,1216])               # sensor size(in mm)
#image_size = np.array([11.345,7.126])
pixel_width = 5.86e-3                       # pixel size in mm
PX= image_size[0]/2.0                       # principal point x-coordinate
PY= image_size[1]/2.0                       # principal point y-coordinate
IMAGE_HEIGTH = image_size[1]
IMAGE_WIDTH = image_size[0]
THETA_X = 0                                 # roll angle
THETA_Y = 0                                 # pitch angle
THETA_Z = 0                                 # yaw angle

# camera Right
THETA_X_R = 0                                 # roll angle
THETA_Y_R = 0                                 # pitch angle
THETA_Z_R= 0                                 # yaw angle
# camera Left
THETA_X_L = 0                                 # roll angle
THETA_Y_L = 0                                 # pitch angle
THETA_Z_L= 0                                 # yaw angle

C_L = np.array([0,0,0])                     # camera centre
C_R = np.array([500,0,0])

chess_dimx,chess_dimy = (13,9)
chess_sq_size = 44

radius = 5