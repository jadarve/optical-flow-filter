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

"""forward difference operator in X (column)"""
_dxp_k = np.array([[1.0, -1.0, 0.0]], dtype=np.float32)

"""backward difference operator in X (column)"""
_dxm_k = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)

"""central difference in X (column)"""
_dxc_k = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)


"""forward difference operator in Y (row)"""
_dyp_k = np.copy(_dxp_k.T)

"""backward difference operator in Y (row)"""
_dym_k = np.copy(_dxm_k.T)

"""central difference in Y (row)"""
_dyc_k = np.copy(_dxc_k.T)


"""+1 shift operator in X (column)"""
_sxp_k = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

"""-1 shift operator in X (column)"""
_sxm_k = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)


"""+1 shift operator in Y (row)"""
_syp_k = np.copy(_sxp_k.T)

"""-1 shift operator in Y (row)"""
_sym_k = np.copy(_sxm_k.T)


def dominantFlow_x(flow_x):
    """Computes dominant flow in X (column) direction.

    Parameters
    ----------
    flow_x : ndarray.
        Optical X flow component.

    Returns
    -------
    flow_x_dom : ndarray.
        Dominant flow in X (column) direction
    """
    
    flow_x_dom = np.copy(flow_x)

    # central difference of absolute value of flow
    flow_x_abs = nd.convolve(np.abs(flow_x), _dxc_k)

    # assign true to pixels with positive absolute difference
    dabs_p = flow_x_abs >= 0

    flow_x_dom[dabs_p] = nd.convolve(flow_x, _sxp_k)[dabs_p]
    flow_x_dom[not dabs_p] = nd.convolve(flow_x, _sxm_k)[not dabs_p]

    return flow_x_dom


def dominantFlow_y(flow_y):
    """Computes dominant flow in Y (row) direction

    Parameters
    ----------
    flow_y : ndarray.
        Optical flow Y component.

    Returns
    -------
    flow_y_dom : ndarray.
        Dominant flow in Y (row) directions.
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


def propagate(flow, iterations=1, dx=1.0, border=3):
    pass


def propagateWithPayload(flow, iterations=1, dx=1.0, border=3, payload=None):
    pass
   

def propagation_step(flow, dx=1.0, dt=1.0, border=3, payload=None):
    """Performs one iteration of the propagation numerical scheme.

    Parameters
    ----------
    flow : ndarray
        Optical flow field. Each pixel (i, j) contains the (u, v)
        components of optical flow.

    dx : float, optional.
        Pixel size. Defaults to 1.0.

    dt : float, optional.
        Time step. Defaults to 1.0.

    border: integer, optional
        Border width in which the propagation does not take place.
        The returned propagated flow with have the same values as
        the input in the border regions. Defaults to 3.

    Returns
    -------
    flowPropagated : 3D ndarray.
        Propagated flow field.

    payloadPropagated: list.
        Propagated payloads or None if payload parameters is None

    Raises
    ------
    ValueError : if flow.ndim != 3
    """

    if flow.ndim != 3:
        raise ValueError('flow field must be a 3D ndarray')
    
    # U V flow components
    U = np.copy(flow[:,:,0])
    V = np.copy(flow[:,:,1])
    
    # ratio between time and pixel size
    R = dt/dx
    
    
    #############################################
    # PROPAGATION IN X (column) DIRECTION
    #
    # Uh = U - R*U*dx(U)
    # Vh = V - R*U*dx(V)
    #############################################

    # dominant flow
    Ud = dominantFlow_x(U)
    
    # sign of Ud
    Up = Ud >= 0
    Um = Ud < 0

    Uh = np.copy(U)
    Vh = np.copy(V)
    
    # propagation with upwind kernels
    Uh[Up] -= R*(Ud*nd.convolve(U, _dxm_k))[Up]
    Uh[Um] -= R*(Ud*nd.convolve(U, _dxp_k))[Um]
    
    Vh[Up] -= R*(Ud*nd.convolve(V, _dxm_k))[Up]
    Vh[Um] -= R*(Ud*nd.convolve(V, _dxp_k))[Um]
    
    # payload propagation
    if payload != None:
        payloadPropX = list()
        
        # for each field in the payload list
        for field in payload:
            
            fieldPropX = np.copy(field)
            fieldPropX[Up] -= R*(Ud*nd.convolve(field, _dxm_k))[Up]
            fieldPropX[Um] -= R*(Ud*nd.convolve(field, _dxp_k))[Um]
            payloadPropX.append(fieldPropX)
    
    
    #############################################
    # PROPAGATION IN Y DIRECTION
    #
    # U1 = Uh - R*Uh*dy(U)
    # V1 = Vh - R*Vh*dy(V)
    #############################################

    Vd = dominantFlow_y(Vh)
    
    # sign of Vd
    Vp = Vd >= 0
    Vm = Vd < 0
    
    U1 = np.copy(Uh)
    V1 = np.copy(Vh)

    U1[Vp] -= R*(Vd*nd.convolve(Uh, _dym_k))[Vp]
    U1[Vm] -= R*(Vd*nd.convolve(Uh, _dyp_k))[Vm]
    
    V1[Vp] -= R*(Vd*nd.convolve(Vh, _dym_k))[Vp]
    V1[Vm] -= R*(Vd*nd.convolve(Vh, _dyp_k))[Vm]
    
    # payload propagation
    payloadPropagated = None
    if payload != None:
        
        payloadPropagated = list()
        
        # for each scalar field in the payload
        for i in xrange(len(payloadPropX)):
            
            field = payloadPropX[i]
            fieldPropY = np.copy(field)
            fieldPropY[Vp] -= R*(Vd*nd.convolve(field, _dym_k))[Vp]
            fieldPropY[Vm] -= R*(Vd*nd.convolve(field, _dyp_k))[Vm]
            
            payloadPropagated.append(fieldPropY)
    
    
    ##############################################
    # PACK THE PROPAGATED FLOW WITH BORDER REMOVAL
    ##############################################
    flowPropagated = np.copy(flow)

    # assign the propagated flow to the interior region of the field
    flowPropagated[border:-border, border:-border, 0] = U1[border:-border, border:-border]
    flowPropagated[border:-border, border:-border, 1] = V1[border:-border, border:-border]
    
    # sanity check
    if np.isnan(flowPropagated).any() or np.isinf(flowPropagated).any():
        print('propagation_step(): NaN or Inf detected in propagated flow')
    
    return flowPropagated, payloadPropagated
