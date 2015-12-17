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


__all__ = ['brightnessModel']


def brightnessModel(img, support=5):
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
        for _ in xrange(support-2):
            blur1D = np.convolve(blur1D, b, mode='full')

        blur1D = np.reshape(blur1D, (1, blur1D.shape[0]))
        gradient1D = np.arange(-support/2 + 1, support/2 + 1, dtype=np.float)
        gradient1D = np.reshape(gradient1D[::-1], (1, gradient1D.shape[0]))

    # renormalize masks
    gradient1D /= np.sum(gradient1D*gradient1D)
    blur1D /= np.sum(blur1D)

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



def updatePyramid(imgPyr, imgOldPyr, flowOldPyr, levels,
                  iterationList, support=3, border=3, gamma=1.0,
                  smoothIterations=1, maxflow=1, mode='reflect',
                  propagate=True, dd=None):
    """Update optical flow
    """
    
    # computes optical deltaFlow at the top level
    A0Top = imgOldPyr[levels -1]
    imgTop = imgPyr[levels -1]
    flowTop = flowOldPyr[levels -1]
    
    A0Pyr = misc.zerosListLike(imgOldPyr)
    flowUpdPyr = misc.zerosListLike(flowOldPyr)
    
    # propagate flowTop
    N = iterationList[levels -1]
    dx = math.pow(2, levels -1)
    
    if not propagate:
        flowTopProp = np.copy(flowTop)
    else:
        flowTopProp = filterpropagate.iterate(flowTop, N, dx, border)
    
    # update flowTop
    flowTopUpd, A0Top, _ = update(imgTop, A0Top, flowTopProp, support, gamma[levels-1], mode)
    for _ in xrange(smoothIterations[levels-1]):
        flowTopUpd = smoothFlow(flowTopUpd, support)
    
    # store flowTopUpd in the output
    flowUpdPyr[levels -1][:] = flowTopUpd[:]
    A0Pyr[levels -1][:] = A0Top[:]
    
    # for the remaining levels of the pyramid
    for h in xrange(levels -2, -1, -1):
        
        # data for this level
        img = imgPyr[h]
        deltaFlow = flowOldPyr[h]
        A0 = imgOldPyr[h]
        N = iterationList[h]
        dx = math.pow(2, h)
        
        # FIXME: revise sanity checks
        # flow field before propagation at this level
        flowLevel = misc.flowUp(flowTop) + deltaFlow
        flowLevel[flowLevel > 20] = 20
        flowLevel[flowLevel < -20] = -20
        
        # transport deltaFlow field and A0 of this level by flowLevel
        payload = [np.copy(deltaFlow[:,:,0]), np.copy(deltaFlow[:,:,1]), A0]
        
        # FIXME: remove propagate flag
        if not propagate:
            _, payloadProp = flowLevel, payload
        else:
            _, payloadProp = filterpropagate.iterateWithPayload(flowLevel, N, dx, border, payload)
        
        # propagated payloads
        deltaFlowProp = np.zeros_like(deltaFlow)
        deltaFlowProp[:,:,0] = payloadProp[0][:,:]
        deltaFlowProp[:,:,1] = payloadProp[1][:,:]
        A0Prop = payloadProp[2]
        
        # update deltaFlow for this level
        dflowUpd, A0, _ = update(img, A0Prop, deltaFlowProp, support, gamma[h], mode)
        
        # border replacement for the delta flow state
        deltaFlowUpd = np.copy(deltaFlowProp)
        deltaFlowUpd[border:-border, border:-border, :] = dflowUpd[border:-border, border:-border, :]
        deltaFlowUpd[deltaFlowUpd > maxflow[h]] = maxflow[h]
        deltaFlowUpd[deltaFlowUpd < -maxflow[h]] = -maxflow[h]
        
        for _ in xrange(smoothIterations[h]):
            deltaFlowUpd = smoothFlow(deltaFlowUpd, support)
        
        flowUpdPyr[h][:] = deltaFlowUpd[:]
        A0Pyr[h][:] = A0[:]
        
        flowTop = flowLevel
    
    if dd != None:
        dd['flowSum'] = np.copy(flowTop)
    
    return flowUpdPyr, A0Pyr
