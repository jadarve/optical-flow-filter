"""
    flowfilter.rotation
    ----------------------

    Module containing rotational optical flow methods.

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import numpy as np
import numpy.linalg as la


def rotationalOpticalFlow(K, ishape, w):
    """Computes rotational optical flow given camera intrinsics and angular velocity.

    Parameters
    ----------
    K : ndarray.
        3x3 camera intrinsics matrix.

    ishape : 2-vector.
        image resolution (height, width).

    w : 3-vector.
        angular velocity (wx, wy, wz).

    Returns
    -------
    phi : ndarray.
        Rotational optical flow field. (height, width, 2) float32 array.
    """

    # inverse intrinsics matrix
    Kinv = la.inv(K)

    # angular velocity components
    wx, wy, wz = w

    # pixel coordinates
    X, Y = np.meshgrid(np.arange(ishape[1]), np.arange(ishape[0]))

    # image plane coordinates
    pcoord = np.zeros((ishape[0], ishape[1], 3))

    # p = Kinv*(x, y, 1)^T
    pcoord[...,0] = Kinv[0,0]*X + Kinv[0,1]*Y + Kinv[0,2]*1
    pcoord[...,1] = Kinv[1,0]*X + Kinv[1,1]*Y + Kinv[1,2]*1
    pcoord[...,2] = Kinv[2,0]*X + Kinv[2,1]*Y + Kinv[2,2]*1

    # cross product between angular velocity and pcoord
    crossProd = np.zeros((ishape[0], ishape[1], 3))
    crossProd[...,0] = -pcoord[...,1]*wz + pcoord[...,2]*wy
    crossProd[...,1] =  pcoord[...,0]*wz - pcoord[...,2]*wx
    crossProd[...,2] = -pcoord[...,0]*wy + pcoord[...,1]*wx

    # rotational optical flow
    phi = np.zeros((ishape[0], ishape[1], 2), dtype=np.float32)
    phi[...,0] = K[0,0]*crossProd[...,0] + (K[0,2] - X)*crossProd[...,2]
    phi[...,1] = K[1,1]*crossProd[...,1] + (K[1,2] - Y)*crossProd[...,2]

    return phi
