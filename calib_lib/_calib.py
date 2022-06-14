"""
calib library. Part of calib_lib package.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
31/04/2022: file creation
-----------------------
09/06/2022 added calib()
14/06/2022 added create_4_point_chess()
"""

import numpy as np
import cv2

from ._variables_pixel import *

from ._checkerboard import createcircle_3d,create_chessboard_1024points,get_chessboard_rot_trans

from ._homogeneous import to_homogeneous,to_inhomogeneous,to_homogeneus_arr

def _get_roll_matrix(theta_x: float = 0.0) -> np.ndarray:
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta_x), -np.sin(theta_x)],
            [0.0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )
    return Rx


def _get_pitch_matrix(theta_y: float = 0.0) -> np.ndarray:
    Ry = np.array(
        [
            [np.cos(theta_y), 0.0, np.sin(theta_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta_y), 0.0, np.cos(theta_y)],
        ]
    )
    return Ry


def _get_yaw_matrix(theta_z: float = 0.0) -> np.ndarray:
    Rz = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0.0],
            [np.sin(theta_z), np.cos(theta_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return Rz


#my and mx correspond to the pixel density in the x and y direction
def get_calibration_matrix(
    f: float,
    px: float = 0.0,
    py: float = 0.0,
    mx: float = 1.0,
    my: float = 1.0,
) -> np.ndarray:
    """
    Returns the calibration matrix

    Parameters
    ----------
    f : float
        focal length
    mx : float
        pixel density in the x diretion
    my : float
        pixel density in the y diretion
    px : float
        principal point in the x diretion
    py : float
        principal point in the y diretion

    Returns
    -------
    K : np.ndarray
        calibration matrix
    """
    K = np.array([[f*mx, 0.0, px], [0.0, f*my, py], [0.0, 0.0, 1.0]])
    return K


def get_rotation_matrix(
    theta_x: float = 0.0, theta_y: float = 0.0, theta_z: float = 0.0
) -> np.ndarray:
    """
    Returns the rotation matrix

    Parameters
    ----------
    theta_x : float
        angle in the x axis in radians
    theta_y : float
        angle in the y axis in radians
    theta_z : float
        angle in the z axis in radians
    Returns
    -------
    rotation_matrix : np.ndarray
        rotation matrix
    """
    # Roll
    Rx = _get_roll_matrix(theta_x)
    # Pitch
    Ry = _get_pitch_matrix(theta_y)
    # Yaw
    Rz = _get_yaw_matrix(theta_z)

    rotation_matrix =  Rz @ Ry @ Rx

    return rotation_matrix

def get_translation_matrix(Tx: float,Ty: float,Tz: float):
    """
    Returns the translation matrix

    Parameters
    ----------
    Tx : float
        translation on the x axis
    Ty : float
        translation on the y axis
    Tz : float
        translation on the z axis
    Returns
    -------
    translation_matrix : np.array
        translation_matrix
    """
    translation_matrix = np.array([[Tx],[Ty],[Tz]])
    return translation_matrix

def get_extrinsic_matrix(rot: np.array((3,3)), trans:np.array((3,1))):
    """
    Returns the extrinsic matrix

    Parameters
    ----------
    rot : np.array
        rotation matrix
    trans : np.array
        translation matrix
    Returns
    -------
    E : np.array
        Extrinsic matrix 
    """

    E = np.concatenate((rot,trans), axis = 1)
    return E

def get_projection_matrix(
    f: float,
    px: float = 0.0,
    py: float = 0.0,
    tx : float = 0.0,
    ty : float = 0.0,
    tz : float = 0.0,
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    theta_z: float = 0.0,
    mx: float = 1.0,
    my: float = 1.0,
) -> np.ndarray:
    """
    Returns the projection matrix

    Parameters
    ----------

    f : float
        focal length
    mx : float
        pixel density in the x diretion
    my : float
        pixel density in the y diretion
    px : float
        principal point in the x diretion
    py : float
        principal point in the y diretion
    tx : float
       translation on the x axis
    ty : float
       translation on the y axis
    tz : float
        translation on the z axis
    theta_x : float
        angle in the x axis in radians
    theta_y : float
        angle in the y axis in radians
    theta_z : float
        angle in the z axis in radians
    mx : float
        pixel density in the x diretion
    my : float
        pixel density in the y diretion
    Returns
    -------
    P : np.array
        Projection matrix 
    """

    K = get_calibration_matrix(f=f, px=px, py=py, mx=mx, my=my)
    #print("Calibration matrix: \n",K)
    R = get_rotation_matrix(theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)
    #print("Rotation matrix: \n",R)
    T = get_translation_matrix(tx,ty,tz)
    #print("Translation matrix: \n",T)
    E = get_extrinsic_matrix(rot = R,trans = T)
    P = K @ E
    
    return P, E

def get_plucker_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Get the plucker matrix

    Parameters
    ----------
    A : np.array
        camera origin 
    B : np.array
        point in the world
    Returns
    -------
    L : np.array
        Plucker matrix
    """
    A = to_homogeneous(A)
    B = to_homogeneous(B)
    L = A.reshape(-1, 1) * B.reshape(1, -1) - B.reshape(-1, 1) * A.reshape(1, -1)
    return L



def get_image_points(rand_points,px,py,thetax = 0, thetay = 0, thetaz = 0,trans_x = 0,trans_y = 0,trans_z = 0,F = 3.0,mx = 1,my = 1):
    """
    from the world points, obtains the correspondent image coordinate

    Parameters
    ----------
    rand_points:  np.array
        world points
    F : float
        focal length
    px : float
        principal point in the x diretion
    py : float
        principal point in the y diretion
    trans_x : float
       translation on the x axis
    trans_y : float
       translation on the y axis
    trans_z : float
        translation on the z axis
    thetax : float
        angle in the x axis in radians
    thetay : float
        angle in the y axis in radians
    thetaz : float
        angle in the z axis in radians
    mx : float
        pixel density in the x diretion
    my : float
        pixel density in the y diretion
    Returns
    -------
    x_arr:    np.array
        image coordinates
    rand_points : np.array
        world coordinates
    """
    K = get_calibration_matrix(F, px=px, py=py,mx=mx,my=my)                                             # calibration matrix
    P, E = get_projection_matrix(F, px=px,py =py,theta_x = thetax,theta_y = thetay,theta_z = thetaz,tx = trans_x,ty = trans_y, tz = trans_z,mx = mx,my =my)                        # projection matrix
    #print("\nCalibration matrix (K):\n", K)
    #print("\nProjection matrix (P):\n", P)

    x = []

    for i in range(rand_points.shape[0]):
        Xh = to_homogeneous(rand_points[i,:])
        xh = P @ Xh
        Xx = to_inhomogeneous(xh)
        x.append(Xx)

    x_arr = np.array(x)

    return x_arr, rand_points, E, K, P


def calibration(world_pts_arr,x_zhang_R,x_zhang_L,image_size):
    """
    From the image plane coordinates in 2D calibrates both cameras and then calibrates the stereo pair
    Parameters
    ----------
    world_pts_arr : np.array
        world coordinates
    x_zhang_R:    np.array
        right image coordinates
    x_zhang_L:    np.array
        Left image coordinates
    image_size : list
        image size
    Returns
    -------
    ret_R:
        error in the Right camera
    mtx_r:
        calibration matrix right camera
    ret_L:
        error in the left camera
    mtx_l:
        calibration matrix left camera
    retS:
        error in the stereo pair
    new_mtxR:
        new calibration matrix right camera
    new_mtxL:
        new calibration matrix left camera
    Rot:
        rotation matrix
    Trns:
        translation matrix
    Emat:
        Essencial matrix
    Fmat:
        Fundamental matrix
    """    
    ret_R, mtx_R, dist_R, rvecs_R, tvecs_R = cv2.calibrateCamera(world_pts_arr, x_zhang_R, (image_size[0],image_size[1]), None, None)
    ret_L, mtx_L, dist_L, rvecs_L, tvecs_L = cv2.calibrateCamera(world_pts_arr, x_zhang_L, (image_size[0],image_size[1]), None, None)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same 

    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(world_pts_arr, x_zhang_L, x_zhang_R, mtx_L, dist_L, mtx_R, dist_R, (image_size[0],image_size[1]), criteria_stereo, flags)

    return ret_R,mtx_R,ret_L,mtx_L,retS,new_mtxL,new_mtxR,Rot,Trns,Emat,Fmat



def create_1024_point_chess(xrange,yrange,zrange,chess_dimx = chess_dimx,chess_dimy = chess_dimy,chess_sq_size = chess_sq_size,radius = radius):

    translst= []
    chess_pts = []
    rotangles = [[np.pi/4,0,np.pi/2]]

    for z in zrange:
        for x in xrange:
            for y in yrange:
                translst.append([x,y,z])

    spots_array,cx0,cy0,r0 = create_chessboard_1024points(chess_dimx,chess_dimy,chess_sq_size,radius,gridSize)


    for trans in translst:
            for rot in rotangles:
                    chess_pts.append(get_chessboard_rot_trans(spots_array,rot[0],rot[1],rot[2],trans[0],trans[1],trans[2]))

    chess_pts_arr = np.array(chess_pts)
    print(chess_pts_arr.shape)
    pattern_size_x,pattern_size_y = (chess_dimx,chess_dimy)

    X_pattern = np.linspace(0, pattern_size_x,pattern_size_x + 1)*chess_sq_size
    Y_pattern = np.linspace(0, pattern_size_y,pattern_size_y + 1)*chess_sq_size
    zdata = np.zeros((pattern_size_x + 1,pattern_size_y + 1))
    xdata, ydata = np.meshgrid(X_pattern, Y_pattern)

    xdata_ = xdata.flatten()
    ydata_ = ydata.flatten()
    zdata_ = zdata.flatten()
    #
    circle_x = np.zeros((xdata_.shape[0],gridSize))
    circle_y = np.zeros((xdata_.shape[0],gridSize))
    circle_z = np.zeros((xdata_.shape[0],gridSize))
    circle_ones = np.ones((xdata_.shape[0],gridSize))

    spots_arr = np.zeros((xdata_.shape[0],gridSize,4))

    cx0 = np.zeros((xdata_.shape[0]))
    cy0 = np.zeros((xdata_.shape[0]))
    r0 = np.zeros((xdata_.shape[0]))

    for coord_i ,(x_d,y_d) in enumerate(zip(xdata_,ydata_)):
        circle_x[coord_i],circle_y[coord_i],cx0[coord_i],cy0[coord_i],r0[coord_i] = createcircle_3d(radius,x_d,y_d,1024)
        spots_array[coord_i,:,:] = np.array([circle_x[coord_i],circle_y[coord_i],circle_z[coord_i],circle_ones[coord_i]]).T

    print(np.shape(spots_array))

    spots = []

    world_pts_arr = np.zeros((chess_pts_arr.shape[0],(pattern_size_x+1)*(pattern_size_y+1),chess_pts_arr.shape[2],3),np.float32)

    for i in range(chess_pts_arr.shape[0]):
        world_pts_arr[i,:,:,:] = spots_array[:,:,:3]

    print(np.shape(world_pts_arr))

    world_pts_arr = world_pts_arr.reshape((chess_pts_arr.shape[0],1,(pattern_size_x+1)*(pattern_size_y+1)*chess_pts_arr.shape[2],3))

    x_zhang_R = np.zeros((chess_pts_arr.shape[0],(pattern_size_x+1)*(pattern_size_y+1)*chess_pts_arr.shape[2],2),np.float32)
    x_zhang_L = np.zeros((chess_pts_arr.shape[0],(pattern_size_x+1)*(pattern_size_y+1)*chess_pts_arr.shape[2],2),np.float32)
    x_arr_L = np.zeros(((pattern_size_x+1)*(pattern_size_y+1),chess_pts_arr.shape[2],2))
    x_arr_R = np.zeros(((pattern_size_x+1)*(pattern_size_y+1),chess_pts_arr.shape[2],2))
    for i in range(chess_pts_arr.shape[0]):
        for j in range(chess_pts_arr.shape[1]):
            x_R, X_R, E_R, K_R, P_R = get_image_points(chess_pts_arr[i,j,:],PX,PY,thetax= 0,thetay = 0,thetaz = 0,trans_x= -C_R[0],trans_y= -C_R[1],trans_z= -C_R[2],F = F,mx = (1/pixel_width),my =(1/pixel_width))
            x_L, X_L, E_L, K_L, P_L = get_image_points(chess_pts_arr[i,j,:],PX,PY,thetax= 0,thetay = 0,thetaz = 0,trans_x= -C_L[0],trans_y= -C_L[1],trans_z= -C_L[2],F = F,mx = (1/pixel_width),my =(1/pixel_width))
        
            x_arr_L[j,:,:] = x_L
            x_arr_R[j,:,:] = x_R

        
        x_zhang_R[i,:,:] = x_arr_R.reshape((pattern_size_x+1)*(pattern_size_y+1)*chess_pts_arr.shape[2],2)
        x_zhang_L[i,:,:] = x_arr_L.reshape((pattern_size_x+1)*(pattern_size_y+1)*chess_pts_arr.shape[2],2)


    x_zhang_R = x_zhang_R.reshape((chess_pts_arr.shape[0],(pattern_size_x+1)*(pattern_size_y+1),chess_pts_arr.shape[2],2))
    x_zhang_L = x_zhang_L.reshape((chess_pts_arr.shape[0],(pattern_size_x+1)*(pattern_size_y+1),chess_pts_arr.shape[2],2))



    print("World points: ",world_pts_arr.shape)
    print("Image points: ",x_zhang_L.shape)

    return world_pts_arr,x_zhang_R,x_zhang_L,P_R,P_L  