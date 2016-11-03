"""
    flowfilter.misc
    ---------------

    Miscelaneous functions.

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import numpy as np
import numpy.linalg as la
import scipy.ndimage as nd

__all__ = ['imagePyramid', 'imageDown', 'imageUp']


"""1D smoothing mask"""
_sx_k = 0.25*np.array([[1.0, 2.0, 1.0]])

"""2D smoothing mask"""
_smooth_k = np.dot(_sx_k.T, _sx_k)


def imagePyramid(img, levels):
    """Creates an image pyramid

    Parameters
    ----------
    img : ndarray
        Image array. It can be a 2D or 3D array. If it is a 3D array,
        the smoothing is applied independently to each channel.

    levels : integer
        Number of levels of the pyramid

    Returns
    -------
    pyr : list[ndarray]
        Image pyramid. pyr[0] contains image at original resolution.
        pyr[h] contains an image subsampled by a factor 2^h.
    """
    
    if levels < 1: raise ValueError('levels should be greater or equal 1')

    if levels == 1:
        return [np.copy(img)]

    else:
        pyr = list()

        # append the first level of the pyramid
        imgDownsampled = np.copy(img)
        pyr.append(imgDownsampled)
        
        for _ in range(levels-1):
            imgDownsampled = imageDown(imgDownsampled)
            pyr.append(imgDownsampled)

        return pyr


def imageDown(img):
    """Downsamples an image by a factor of 2.

    First, input image is convolved with the following
    smoothing mask

         [ 1.  2.  1.]
    1/16*[ 2.  4.  2.]
         [ 1.  2.  1.]

    Next, the smoothed image is downsampled by a factor of 2.

    Parameters
    ----------
    img : ndarray
        Image array. It can be a 2D or 3D array. If it is a 3D array,
        the smoothing is applied independently to each channel.

    Returns
    -------
    imdown : ndarray
        Downsampled image.


    See also
    --------
    imagePyramid : Creates an image pyramid.
    imageUp : Upsample input image by a factor of 2.
    """
    
    if img.ndim == 2:
        
        # smooth and downsample by 2
        imdown = nd.convolve(img, _smooth_k)
        return np.copy(imdown[::2, ::2])

    else:
        depth = img.shape[2]

        # smooth and downsample each img component
        smoothList = [nd.convolve(img[...,n], _smooth_k)[::2, ::2] for n in range(depth)]

        # recombine channels and return
        return np.concatenate([p[...,np.newaxis] for p in smoothList], axis=2)


def imageUp(img, order=1):
    """Upsample input image by a factor of 2.

    Parameters
    ----------
    img : ndarray
        Image array. It can be a 2D or 3D array. If it is a 3D array,
        the smoothing is applied independently to each channel.

    order : integer, optional
        Interpolation order. Defaults to 1

    Returns :
    imgUp : ndarray
        Upsampled image of size (2*H, 2*W, D) where (H, W, D) is the
        width, height and depth of the input image
    """
    
    if img.ndim == 2:
        imgZoomed = np.zeros([2*img.shape[0], 2*img.shape[1]], dtype=img.dtype)
        nd.zoom(img, 2.0, output=imgZoomed, order=order, mode='reflect')
        return imgZoomed

    else:

        zoomList = list()
        for d in range(img.shape[2]):

            imgZoomed = np.zeros([2*img.shape[0], 2*img.shape[1]], dtype=img.dtype)
            nd.zoom(img[...,d], 2.0, output=imgZoomed, order=order, mode='reflect')

            zoomList.append(imgZoomed)

        # recombine channels and return
        return np.concatenate([p[...,np.newaxis] for p in zoomList], axis=2)


def endpointError(flow1, flow2):
    """returns Endpoint Error between two flow fields

    Parameters
    ----------
    flow1 : ndarray.
        First optical flow field.

    flow2 : ndarray.
        Second optical flow field.

    Returns
    -------
    EE : endpoint error field.
        Scalar field with the endpoint error.
    """

    return la.norm(flow1 - flow2, axis=2)


def angularError(flow1, flow2):
    """returns the angular error between two flow fields.

    Parameters
    ----------
    flow1 : ndarray.
        First optical flow field.

    flow2 : ndarray.
        Second optical flow field.

    Returns
    -------
    AE : angular error field.
        Scalar field with the angular error field in degrees.
    """
    
    f1_x = flow1[...,0]
    f1_y = flow1[...,1]
    
    f2_x = flow2[...,0]
    f2_y = flow2[...,1]
    
    top = 1.0 + f1_x*f2_x + f1_y*f2_y
    bottom = np.sqrt(1.0 + f1_x*f1_x + f1_y*f1_y)*np.sqrt(1.0 + f2_x*f2_x + f2_y*f2_y)
    return np.rad2deg(np.arccos(top / bottom))
