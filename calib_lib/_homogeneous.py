"""
Homogeneous library. Part of calib_lib package.
Developed by João Alegria @ Aveiro University, Portugal, 2022.
email : joaoalegria31@ua.pt
Msc Thesis

-----------------------
31/10/2022: file creation
"""

import numpy as np
from calib_lib import *


def to_inhomogeneous(X: np.ndarray) -> np.ndarray:
    """Turns a np.ndarray into inhomogeneous

    Parameters
    ----------
    X : np.ndarray
        Homogeneous array
    Returns
    -------
    X: np.ndarray
        Inhomogeneous array
    """

    if X.ndim > 1:
        raise ValueError("x must be one-dimensional.")

    return (X / X[-1])[:-1]


def to_homogeneous(X: np.ndarray) -> np.ndarray:
    """Turns a np.ndarray into homogeneous

    Parameters
    ----------
    X : np.ndarray
        Inhomogeneous array
    Returns
    -------
    X: np.ndarray
        Homogeneous array
    """

    if X.ndim > 1:
        raise ValueError("X must be one-dimensional.")

    return np.hstack([X, 1])

def to_homogeneus_arr(X: np.ndarray) -> np.ndarray:
    """Turns a array into homogeneous

    Parameters
    ----------
    X : np.ndarray
        Inhomogeneous array
    Returns
    -------
    outvalue: np.ndarray
        Homogeneous array
    """

    one_matrix = np.ones((X.shape[0],1))

    out_value = np.concatenate((X, one_matrix), axis=1)

    return out_value