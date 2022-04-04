"""
checkerboard library. Part of calib_lib package.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
31/10/2022: file creation
"""

import numpy as np
from calib_lib import *


def create_chessboard(patternsize_x,patternsize_y,dim):
    """
    Creates chessboard with pattern dimensions
    
    Parameters
    ----------
    patternsize_x : int
        size x of the chessboard 
    patternsize_y : int
        size y of the chessboard
    dim:    int
        distance between corners in mm

    Returns
    -------
    world_pts_arr :  : np.array
        chessboard points in 3D
    """
    pattern_size_x,pattern_size_y = (patternsize_x,patternsize_y)

    X = np.linspace(-pattern_size_x/2, pattern_size_x/2,pattern_size_x + 1)*dim
    Y = np.linspace(-pattern_size_y/2, pattern_size_y/2,pattern_size_y + 1)*dim

    zdata = np.zeros((patternsize_x + 1,patternsize_y + 1))
    xdata, ydata = np.meshgrid(X, Y)

    xdata_ = xdata.flatten()
    ydata_ = ydata.flatten()
    zdata_ = zdata.flatten()
    onedata = np.ones((zdata_.shape[0])).flatten()
    # homogeneous coordinates
    world_pts =([xdata_,ydata_,zdata_,onedata])
    world_pts_arr = np.array(world_pts).T

    return world_pts_arr

def get_chessboard_rot_trans(world_pts_arr,rx = 0.0, ry = 0.0, rz = 0.0, tx = 0.0,ty = 0.0,tz = 0.0):
    """
    rotates and translates the chessboard

    Parameters
    ----------
    world_pts_arr:  np.array
        chessboard points in homogenous coordinates[4,1]
    rx: float
        rotation on the x axis
    ry: float
        rotation on the y axis
    rz: float
        rotation on the z axis
    tx: float
        translation on the x axis
    ty: float
        translation on the y axis
    tz: float
        translation on the z axis

    Returns
    -------
    world_trans_arr:    np.array
        output chessboard rotation and translation 
    """ 
    
    Rx =get_rotation_matrix(rx,ry,rz)
    Trans = np.array([[tx],[ty],[tz]])
    E = np.concatenate((Rx,Trans),axis=1)

    world_trans_arr = np.dot(world_pts_arr,E.T)

    return world_trans_arr





def get_plane(X,dimx,dimy):
    """
    Creates the surface to use in the function plot_surface.


    Parameters
    ----------
    
    X:  np.array
        array with the coordinate set
    dimx: int
        dimension of the chessboard along the x axis
    dimy: int
        dimension of the chessboard along the y axis
    Returns
    -------
    x_surface: np.array
        surface plane x
    y_surface: np.array
        surface plane y
    z_surface: np.array
        surface plane z

    """
    x_plane, y_plane, z_plane = X[:,0],X[:,1],X[:,2]
    if(dimx == 8):
        x_surface = np.array( [ [x_plane[0],x_plane[dimx]] ,[x_plane[dimx+1],x_plane[((dimx+1)*2) - 1]] ,[x_plane[(dimx+1)*2],x_plane[((dimx+1)*3) - 1]] ,[x_plane[(dimx+1)*3],x_plane[((dimx+1)*4) - 1]],[x_plane[(dimx+1)*4],x_plane[((dimx+1)*5) - 1]],[x_plane[(dimx+1)*5],x_plane[((dimx+1)*6) - 1]],[x_plane[(dimx+1)*6],x_plane[-1]]])
        y_surface = np.array( [ [y_plane[0],y_plane[dimx]] ,[y_plane[dimx+1],y_plane[((dimx+1)*2) - 1]] ,[y_plane[(dimx+1)*2],y_plane[((dimx+1)*3) - 1]] ,[y_plane[(dimx+1)*3],y_plane[((dimx+1)*4) - 1]],[y_plane[(dimx+1)*4],y_plane[((dimx+1)*5) - 1]],[y_plane[(dimx+1)*5],y_plane[((dimx+1)*6) - 1]],[y_plane[(dimx+1)*6],y_plane[-1]]])
        z_surface = np.array( [ [z_plane[0],z_plane[dimx]] ,[z_plane[dimx+1],z_plane[((dimx+1)*2) - 1]] ,[z_plane[(dimx+1)*2],z_plane[((dimx+1)*3) - 1]] ,[z_plane[(dimx+1)*3],z_plane[((dimx+1)*4) - 1]],[z_plane[(dimx+1)*4],z_plane[((dimx+1)*5) - 1]],[z_plane[(dimx+1)*5],z_plane[((dimx+1)*6) - 1]],[z_plane[(dimx+1)*6],z_plane[-1]]])
    elif(dimx == 4):
        x_surface = np.array( [ [x_plane[0],x_plane[dimx]] ,[x_plane[dimx+1],x_plane[((dimx+1)*2) - 1]] ,[x_plane[(dimx+1)*2],x_plane[((dimx+1)*3) - 1]]])
        y_surface = np.array( [ [y_plane[0],y_plane[dimx]] ,[y_plane[dimx+1],y_plane[((dimx+1)*2) - 1]] ,[y_plane[(dimx+1)*2],y_plane[((dimx+1)*3) - 1]]])
        z_surface = np.array( [ [z_plane[0],z_plane[dimx]] ,[z_plane[dimx+1],z_plane[((dimx+1)*2) - 1]] ,[z_plane[(dimx+1)*2],z_plane[((dimx+1)*3) - 1]]])


    return x_surface, y_surface,z_surface