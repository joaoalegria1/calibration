"""
calib library. Part of calib_lib package.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
31/10/2022: file creation
"""

import numpy as np
from calib_lib import *

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
    K = np.diag([mx, my, 1]) @ np.array([[f, 0.0, px], [0.0, f, py], [0.0, 0.0, 1.0]])
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



def get_image_points(rand_points,px,py,thetax = 0, thetay = 0, thetaz = 0,trans_x = 0,trans_y = 0,trans_z = 0,F = 3.0):
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

    Returns
    -------
    x_arr:    np.array
        image coordinates
    rand_points : np.array
        world coordinates
    """
    K = get_calibration_matrix(F, px=px, py=py)                                             # calibration matrix
    P, E = get_projection_matrix(F, px=px,py =py,theta_x = thetax,theta_y = thetay,theta_z = thetaz,tx = trans_x,ty = trans_y, tz = trans_z)                        # projection matrix
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

