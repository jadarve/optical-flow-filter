"""
    flowfilter.misc
    ---------------

    Miscelaneous functions.

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import numpy as np
import scipy.ndimage as nd

__all__ = []


"""1D smoothing mask"""
_sx_k = 0.25*np.array([[1.0, 2.0, 1.0]])

"""2D smoothing mask"""
_smooth_k = np.dot(_sx_k.T, _sx_k)


def imagePyramid(img, levels):
    """Creates an image pyramid
    """
    
    if levels < 1: raise ValueError('levels should be greater or equal 1')

    if levels == 1:
        return np.copy(img)

    else:
        pyr = list()

        # append the first level of the pyramid
        imgDownsampled = np.copy(img)
        pyr.append(imgDownsampled)
        
        for _ in xrange(levels):
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
        imdown = ndimage.convolve(img, _smooth_k)
        return np.copy(imdown[::2, ::2])

    else:
        depth = img.shape[2]

        # smooth and downsample each img component
        smoothList = [nd.convolve(img[...,n])[::2, ::2] for n in range(depth)]

        # recombine channels and return
        return np.concatenate([p[...,np.newaxis] for p in smoothList])


def imageUp(img, order=1):
    """Upsample input image by a factor of 2.

    """
    
    imgZoomed = np.zeros([2*img.shape[0], 2*img.shape[1]], dtype=img.dtype)
    ndimage.zoom(img, 2.0, output=imgZoomed, order=order, mode='reflect')
    
    return imgZoomed
