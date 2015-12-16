"""
    flowfilter.propagation
    ----------------

    Module containing propagation methods.

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import numpy as np
import scipy.ndimage as nd


__all__ = ['dominantFlow_x', 'dominantFlow_y']


###########################################################
# GLOBAL VARIABLES
###########################################################

# forward difference kernels in x and y
_dxp_k = np.array([[1.0, -1.0, 0.0]])
_dyp_k = np.copy(_dxp_k.T)

# backward difference kernels in x and y
_dxm_k = np.array([[0.0, 1.0, -1.0]])
_dym_k = np.copy(_dxm_k.T)

# central difference kernels in x and y
_dxc_k = np.array([[1.0, 0.0, -1.0]])
_dyc_k = np.copy(_dxc_k.T)

# +1 shift operator
_sxp_k = np.array([[1.0, 0.0, 0.0]])
_syp_k = np.copy(_sxp_k.T)

# -1 shift operator
_sxm_k = np.array([[0.0, 0.0, 1.0]])
_sym_k = np.copy(_sxm_k.T)


def dominantFlow_x(flow_x):
    """
    Computes dominant flow in X (column) direction.

    Parameters
    ----------
    flow_x : 2D ndarray.
        Optical flow component in X direction

    Returns
    -------
    flow_x_dom : 2D ndarray.
        Dominant flow in X
    """
    
    flow_x_dom = np.copy(flow_x)

    # central difference of absolute value of flow
    flow_x_abs = nd.convolve(np.abs(flow_x), _dxc_k)

    # assign true to pixels with possitive absolute difference
    dabs_p = flow_x_abs >= 0

    flow_x_dom[dabs_p] = nd.convolve(flow_x, _sxp_k)[dabs_p]
    flow_x_dom[not dabs_p] = nd.convolve(flow_x, _sxm_k)[not dabs_p]

    return flow_x_dom


def dominantFlow_y(flow_y):
    """
    Computes dominant flow in Y (row) direction

    Parameters
    ----------
    flow_y : 2D ndarray.
        Optical flow component in Y (row) direction

    Returns
    -------
    flow_y_dom : ndarray.
        Dominant flow in Y
    """
    
    flow_y_dom = np.copy(flow_y)

    # central difference of absolute value of flow
    flow_y_abs = nd.convolve(np.abs(flow_y), _dyc_k)

    # assign true to pixes with possitive absolute difference
    dabs_p = flow_y_abs >= 0

    # assign possitive or negative shifte
    flow_y_dom[dabs_p] = nd.convolve(flow_y, _syp_k)[dabs_p]
    flow_y_dom[not dabs_p] = nd.convolve(flow_y, _sym_k)[not dabs_p]

    return flow_y_dom

