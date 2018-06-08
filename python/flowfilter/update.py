"""
    flowfilter.update
    -----------------

    Module containing Python implementationso of the filter
    image model and update methods.

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""


import math
import numpy as np
import scipy.ndimage as nd


__all__ = ['imageModel', 'update', 'smoothFlow']


def imageModel(img, support=5):
    """Computes brightness model parameters.

    Parameters
    ----------
    img : ndarray
        Input image in gray scale. If img.dtype is different than
        float32, it is automatically converted.

    support : integer, optional
        Window support used for computing brightness parameters.
        The value should be an odd number greater or equal 3.
        Defaults to 5.

    Returns
    -------
    A0 : ndarray
        Constant brightness term.

    Ax : ndarray
        X (column) gradient component.

    Ay : ndarray
        Y (row) gradient component.

    Raises
    ------
    ValueError : support < 3 or support % 2 != 1:
    """

    if support < 3 or support % 2 != 1:
        raise ValueError('support should be an odd number greater or equal 3')

    # input image dtype check
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    # creates convolution masks
    if support == 3:
        blur1D = np.array([[1.0, 2.0, 1.0]], dtype=np.float32)
        gradient1D = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)

    else:
        b = np.array([1.0, 1.0], dtype=np.float32)
        blur1D = np.array([1.0, 1.0], dtype=np.float32)
        for _ in range(support-2):
            blur1D = np.convolve(blur1D, b, mode='full')

        blur1D = np.reshape(blur1D, (1, blur1D.shape[0]))
        halfSupport = np.floor(support/2)
        gradient1D = np.arange(-halfSupport, halfSupport + 1, dtype=np.float)
        gradient1D = np.reshape(gradient1D[::-1], (1, gradient1D.shape[0]))

    # renormalize masks
    blur1D /= np.sum(blur1D)
    gradient1D *= blur1D

    # Gaussian blurring in X and Y
    imgBlurX = nd.convolve(img, blur1D)
    imgBlurY = nd.convolve(img, blur1D.T)

    # brightness parameters
    Ax = nd.convolve(imgBlurY, gradient1D)
    Ay = nd.convolve(imgBlurX, gradient1D.T)
    A0 = nd.convolve(imgBlurY, blur1D)

    return A0, Ax, Ay


def update(img, imgOld, flowPredicted, support=5, gamma=1.0):
    """Update the optical flow field provided new image data.

    Parameters
    ----------
    img : ndarray
        New brightness image.

    imgOld : ndarray
        Old brightness image. This corresponds to the old

    flowPredicted : ndarray
        Predicted estimation of optical flow. 

    support : integer, optional
        Window support used for computing brightness parameters.
        The value should be an odd number greater or equal 3.
        Defaults to 5.

    gamma : float, optional
        temporal regularization gain controlling the relevance
        of the predicted flow in the update. Value should be
        greater than 0.0. Defaults to 1.0.

    Returns
    -------
    flowUpdated : ndarray
        Updated optical flow field.

    A0 : ndarray
        Constant brightness model parameter computed from
        img.

    Raises
    ------
    ValueError : if gamma <= 0.0
    """

    if gamma <= 0.0: raise ValueError('gamma should be greater than zero')
    
    # compute the image model parameters
    A0, Ax, Ay = imageModel(img, support)
    
    # temporal derivative
    Yt = imgOld - A0
    
    # adjunct matrix N elements for each pixel
    N00 = np.zeros(img.shape); N00[:,:] = gamma + Ay*Ay
    N01 = np.zeros(img.shape); N01[:,:] = -Ax*Ay
    N10 = np.zeros(img.shape); N10[:,:] = np.copy(N01)
    N11 = np.zeros(img.shape); N11[:,:] = gamma + Ax*Ax
    
    # determinant of M for each pixel
    detM = (gamma*(gamma + (Ax*Ax + Ay*Ay)))
    
    # q components for each pixel
    qx = gamma*flowPredicted[:,:,0] + Ax*Yt
    qy = gamma*flowPredicted[:,:,1] + Ay*Yt
    
    # compute the updated optic-flow
    flowX = (N00*qx + N01*qy) / detM
    flowY = (N10*qx + N11*qy) / detM
    
    # pack the results
    flowUpdated = np.concatenate([p[...,np.newaxis] for p in [flowX, flowY]], axis=2)
    
    return flowUpdated, A0


def smoothFlow(flow, iterations=1, support=5):
    """Apply a smoothing filter to optical flow

    Parameters
    ----------
    flow : ndarray
    iterations : integer, optional
    support : integer, optional

    """
    
    if iterations <= 0: raise ValueError('iterations should be greater than 1')
    if support < 3 or support % 2 != 1:
        raise ValueError('support should be an odd number greater or equal 3')

    # average mask
    avg_k = np.ones((support, support), dtype=np.float32) / float(support*support)

    flowSmoothed = np.copy(flow)
    
    for _ in range(iterations):
        # apply smoothing to each flow component    
        for n in range(2):
            flowSmoothed[...,n] = nd.convolve(flowSmoothed[...,n], avg_k)
    
    
    return flowSmoothed