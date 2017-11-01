"""
    flowfilter.plot
    ---------------

    Module containing functions to plot flow fields.

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import pkg_resources

import numpy as np
import scipy.ndimage.interpolation as interp
import scipy.misc as misc


__all__ = ['flowToColor', 'colorWheel']

# load color wheel image
colorWheel = misc.imread(pkg_resources.resource_filename('flowfilter.rsc', 'colorWheel.png'), flatten=False)

# RGB components of colorwheel
_colorWheel_R = np.copy(colorWheel[...,0])
_colorWheel_G = np.copy(colorWheel[...,1])
_colorWheel_B = np.copy(colorWheel[...,2])

def flowToColor(flow, maxflow=1.0):
    """Returns the color wheel encoded version of the flow field.

    Parameters
    ----------
    flow : ndarray
        Optical flow field.

    maxflow : float, optional
        Maximum flow magnitude. Defaults to 1.0.

    Returns
    -------
    flowColor : ndarray
        RGB color encoding of input optical flow.

    Raises
    ------
    ValueError : if maxflow <= 0.0
    """

    if maxflow <= 0.0: raise ValueError('maxflow should be greater than zero')
    
    # height and width of color wheel texture
    h, w = colorWheel.shape[0:2]

    # scale optical flow to lie in range [0, 1]
    flow_scaled = (flow + maxflow) / float(2*maxflow)

    # re-scale to lie in range [0, w) and [0, h)
    flow_scaled[:,:,0] *= (w-1)
    flow_scaled[:,:,1] *= (h-1)

    # reshape to create a list of pixel coordinates
    flow_scaled = np.reshape(flow_scaled, (flow.shape[0]*flow.shape[1], 2)).T

    # swap x, y components of flow to match row, column
    flow_swapped = np.zeros_like(flow_scaled)
    flow_swapped[0,:] = flow_scaled[1,:]
    flow_swapped[1,:] = flow_scaled[0,:]

    # mapped RGB color components
    color_R = np.zeros((flow.shape[0]*flow.shape[1]), dtype=np.uint8)
    color_G = np.zeros_like(color_R)
    color_B = np.zeros_like(color_R)

    # interpolate flow coordinates into RGB textures
    interp.map_coordinates(_colorWheel_R, flow_swapped, color_R, order=0, mode='nearest', cval=0)
    interp.map_coordinates(_colorWheel_G, flow_swapped, color_G, order=0, mode='nearest', cval=0)
    interp.map_coordinates(_colorWheel_B, flow_swapped, color_B, order=0, mode='nearest', cval=0)

    # creates output image
    flowColor = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flowColor[:,:,0] = color_R.reshape(flow.shape[0:2])
    flowColor[:,:,1] = color_G.reshape(flow.shape[0:2])
    flowColor[:,:,2] = color_B.reshape(flow.shape[0:2])
    
    return flowColor

