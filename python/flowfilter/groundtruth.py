"""
    flowfilter.groundtruth
    ----------------------

    Module containing methods to compute ground truth optical flow 

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import numpy as np
import numpy.linalg as la

__all__ = ['rotationalOpticalFlow', 'groundTruthOpticalFlow']


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


def groundTruthOpticalFlow(K, depth, camvel, camangvel):
    """Computes the optical flow for a given scene (depth) and camera velocity.

    The velocity of a point in the scene, expressed in the camera
    reference frame is
    .. math::
        \begin{equation}
            \dot{x} = -Vc - \Omega_\times x
        \end{equation}
    
    Point $x$ projects on the image plane of the camera as
    .. math::
        \begin{equation}
            p = \frac{K x}{e_3^\top x}
        \end{equation}

    The induced optical flow is
    .. math::
        \begin{align}
            \dot{p} &= \frac{d}{t} \left[ \frac{K x}{e_3^\top x} \right] \\
            \dot{p} &= \frac{1}{e_3^\top x} \left[ K \dot{x} - \frac{K x \dotp{e_3}{\dot{x}}}{e_3^\top x}  \right]
        \end{align}

    Parameters
    ----------
    K : 3x3 ndarray.
        Camera intrinsics matrix.

    depth : 2D ndarray.
        Depth map of the scene. Depth at each pixel is the distance
        between the camera center to the point in the scene in that
        specific direction.

    camvel : 3 vector.
        Camera linear velocity in XYZ.

    camangvel : 3 vector.
        Camera angular velocity in radians in XYZ.

    Returns
    -------
    oflow : 3D ndarray.
        Optical flow field.
    """

    # inverse intrinsics matrix
    Kinv = la.inv(K)

    # linear and velocity components
    vx, vy, vz = camvel
    wx, wy, wz = camangvel

    height, width = depth.shape

    # pixel coordinates
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # image plane coordinates
    pcoord = np.zeros((height, width, 3))

    # p = Kinv*(x, y, 1)^T
    pcoord[...,0] = Kinv[0,0]*X + Kinv[0,1]*Y + Kinv[0,2]*1
    pcoord[...,1] = Kinv[1,0]*X + Kinv[1,1]*Y + Kinv[1,2]*1
    pcoord[...,2] = Kinv[2,0]*X + Kinv[2,1]*Y + Kinv[2,2]*1


    # spherical coordinate for each pixel
    etas = np.copy(pcoord)

    pnorm = la.norm(pcoord, axis=2)
    for n in range(3):
        etas[...,n] /= pnorm

    # 3D scene coordinates
    spos = np.copy(etas)
    for n in range(3):
        spos[...,n] *= depth


    ###########################################################
    # 3D scene velocity = -camvel - camangvel cross spos
    ###########################################################
    svel = np.zeros_like(spos)

    # camangvel cross spos
    crossProd = np.zeros((height, width, 3))
    crossProd[...,0] = wy*spos[...,2] - wz*spos[...,1]
    crossProd[...,1] = wz*spos[...,0] - wx*spos[...,2]
    crossProd[...,2] = wx*spos[...,1] - wy*spos[...,0]

    # scene velocity at each point
    svel[...,0] = -vx - crossProd[...,0]
    svel[...,1] = -vy - crossProd[...,1]
    svel[...,2] = -vz - crossProd[...,2]

    
    ###########################################################
    # optical flow
    ###########################################################    
    KtimesSvel = np.zeros_like(spos)     # matrix product [K svel]
    KtimesSpos = np.zeros_like(spos)     # matrix product [K spos]
    for n in range(3):
        KtimesSvel[...,n] = K[n,0]*svel[...,0] + K[n,1]*svel[...,1] + K[n,2]*svel[...,2]
        KtimesSpos[...,n] = K[n,0]*spos[...,0] + K[n,1]*spos[...,1] + K[n,2]*spos[...,2]
        
    # 1 / spos.z
    oneOverZ = 1.0 / spos[...,2]
    
    # Z component of svel
    svelZ = svel[...,2]    
    
    oflow = np.zeros((spos.shape[0], spos.shape[1], 2), dtype=np.float32)
    for n in range(2):
        oflow[...,n] = oneOverZ*(KtimesSvel[...,n] - oneOverZ*svelZ*KtimesSpos[...,n])
        
    return oflow

    