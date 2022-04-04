"""
DLT_lib . Part of calib_lib pacakge.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
31/10/2022: file creation
"""

import numpy as np
from calib_lib import *



def SVD(M):
    """
    Singular value decomposition

    Parameters
    ----------
    M : np.ndarray
        Homography from DLT
    Returns
    -------
    PEst: np.ndarray
        Projective matrix estimation
    """

    U, S, Vh = np.linalg.svd(M)

    L = Vh[-1,:] / Vh[-1,-2]
    #Camera projection matrix:
    PEst = L.reshape(3,3+1)
    #print("Camera Matrix:\n",PEst)

    return PEst


def DLT(x_arr,X_arr):

    worldm, worldn = X_arr.shape
    imagem, imagen = x_arr.shape
    
    #3D DLT
    A = []
    world_pts = X_arr
    image_pts = x_arr

    for i in range(worldm):
        A.append([-world_pts[i,0],-world_pts[i,1],-world_pts[i,2],-1,0,0,0,0,image_pts[i,0]*world_pts[i,0],image_pts[i,0]*world_pts[i,1],image_pts[i,0]*world_pts[i,2],image_pts[i,0]])
        A.append([0,0,0,0,-world_pts[i,0],-world_pts[i,1],-world_pts[i,2],-1,image_pts[i,1]*world_pts[i,0],image_pts[i,1]*world_pts[i,1],image_pts[i,1]*world_pts[i,2],image_pts[i,1]])

    # pass the list A as an array
    M = np.asarray(A).reshape(worldm*2,12)
    #print(M.shape)
    
    PEst = SVD(M)
    return PEst