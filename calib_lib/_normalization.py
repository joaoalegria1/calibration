"""
normalization library. Part of calib_lib package.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
31/10/2022: file creation
"""

import numpy as np
from calib_lib import *


def normalization(data, dim):
    '''
    Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Parameters
    ----------
    data: np.array 
        the data to be normalized (directions at different columns and points at rows) -> (nº points, nº coordinates)
    dim : int
        dimension of the coordinates. If 2 then it's (x,y), if 3 then (x,y,z)

    T : np.array
        the normalized matrix (translation plus scaling)
    coord_n: np.array
        normalized coordinates
    '''

    N = data.shape[0]
    
    if dim == 2:                                                     # 2D points
        x, y = data[:, 0], data[:, 1]                               # x and y coordinates
        x_mean, y_mean = x.mean(), y.mean()                         # mean 
        x_par = (x-x_mean)**2                                       # x part
        y_par = (y-y_mean)**2                                       # y part

        d = (1/N)* np.sum(np.sqrt(x_par + y_par))                   # d
        s_xy = np.sqrt(2)/d                                         # sqrt(2)/d

        T = np.zeros((3,3))                                         # normalization matrix for 2D points
        T[0,0] = s_xy
        T[0,2] = -x_mean*s_xy
        T[1,1] = s_xy
        T[1,2] = -y_mean*s_xy
        T[-1,-1] = 1

        coord_n = np.dot(data,np.transpose(T))                          # normalized image coordinates

    elif(dim== 3):                                                  # 3D points
        X, Y, Z = data[:, 0], data[:, 1], data[:,2]                 # x and y coordinates
        X_mean, Y_mean,Z_mean = X.mean(), Y.mean(),Z.mean()         # mean 
        X_par = (X-X_mean)**2                                       # x part
        Y_par = (Y-Y_mean)**2                                       # y part
        Z_par = (Z-Z_mean)**2                                       # z part

        D = (1/N)* np.sum(np.sqrt(X_par + Y_par + Z_par))           # d
        s_xyz = np.sqrt(3)/D                                        # sqrt(2)/d

        T = np.zeros((4,4))                                         # normalization matrix for 3D points
        T[0,0] = s_xyz
        T[0,3] = -X_mean*s_xyz
        T[1,1] = s_xyz
        T[1,3] = -Y_mean*s_xyz
        T[2,2] = s_xyz
        T[2,3] = -Z_mean*s_xyz
        T[-1,-1] = 1

        coord_n = np.dot(data,T.T)                          # normalized world coordinates

    else:
        raise ValueError('Dataset must be a collection of 2D or points')

    return T,coord_n

def denormalization(H_norm,Txy,Txyz):
    '''
    denormalization matrix

    Parameters
    ----------
    H_norm: np.array 
        projection matrix normalized( from DLT with X and x normalized and homogeneous)
    Txy : np.array
        the normalized matrix for image coordinates
    Txyz : np.array
        the normalized matrix for world coordinates

    H: np.array
        denormalized matrix
    '''
    H = np.linalg.pinv(Txy).dot((H_norm).dot(Txyz))
    H = H / H[-1, -2]

    return H