"""
checkerboard library. Part of calib_lib package.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
31/10/2022: file creation
"""

import numpy as np
from calib_lib import _calib

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

def createcircle_3d(r0,cx0,cy0,nrpoints = 100):
    """
    create the circle, using the function findcircle()

    Parameters
    ----------
    r0 : np.float
        circle radius  
    cx0:    np.float
        centroid x
    cy0:    np.float
        centroid y 
    Returns
    -------
    circle_x:    np.array
        points that define the circle in the cx0 coordinate
    circle_y:    np.array
        points that define the circle in the cy0 coordinate
    cx0:    np.float
        centroid x
    cy0:    np.float
        centroid y    
    r0 : np.float
        circle radius 
  
    """    

    theta = np.linspace(0, 2*np.pi, nrpoints)
    x = r0*np.cos(theta)
    y = r0*np.sin(theta)

    circle_x = x+cx0
    circle_y = y+cy0


    return circle_x, circle_y,cx0,cy0,r0


def create_chessboard_4points(patternsize_x,patternsize_y,dim):
    """
    Creates chessboard with pattern dimensions for pixelization - create 4 points around centroid
    
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
    #
    spots = []
    for i in range(xdata_.shape[0]):
        spots.append(np.array([ [xdata_[i]-5,ydata_[i],zdata_[i],onedata[i]],
                                [xdata_[i],ydata_[i]+5,zdata_[i],onedata[i]],
                                [xdata_[i]+5,ydata_[i],zdata_[i],onedata[i]],
                                [xdata_[i],ydata_[i]-5,zdata_[i],onedata[i]]]))

    spots_array = np.array(spots)
    print(spots_array.shape)

    return spots_array

def create_chessboard_1024points(patternsize_x,patternsize_y,dim,r_0,gridSize):
    """
    Creates chessboard with pattern dimensions for pixelization - create 1024 points around centroid
    
    Parameters
    ----------
    patternsize_x : int
        size x of the chessboard 
    patternsize_y : int
        size y of the chessboard
    dim:    int
        distance between corners in mm
    cx0:    np.float
        centroid x
    cy0:    np.float
        centroid y    
    r0 : np.float
        circle radiu
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
    #
    circle_x = np.zeros((xdata_.shape[0],gridSize))
    circle_y = np.zeros((xdata_.shape[0],gridSize))
    circle_z = np.zeros((xdata_.shape[0],gridSize))
    circle_ones = np.ones((xdata_.shape[0],gridSize))

    spots_array = np.zeros((xdata_.shape[0],4,gridSize))

    cx0 = np.zeros((xdata_.shape[0]))
    cy0 = np.zeros((xdata_.shape[0]))
    r0 = np.zeros((xdata_.shape[0]))
    
    for coord_i ,(x_d,y_d) in enumerate(zip(xdata_,ydata_)):
        circle_x[coord_i],circle_y[coord_i],cx0[coord_i],cy0[coord_i],r0[coord_i] = createcircle_3d(r_0,x_d,y_d,1024)

        spots_array[coord_i,:,:] = [circle_x[coord_i],circle_y[coord_i],circle_z[coord_i],circle_ones[coord_i]]


    spots = []

    for spot in spots_array:
        spots.append(spot.T)

    spots_arr = np.array(spots)

    return spots_arr,cx0,cx0,r0


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
    
    Rx = _calib.get_rotation_matrix(rx,ry,rz)
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