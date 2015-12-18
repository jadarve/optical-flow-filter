"""
    flowfilter.flowfilter
    ---------------------

    Module containing abstract filter classes and Python
    implementation of the optical flow filter algorithm

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import time

import numpy as np


from . import propagation as prop
from . import update as upd


__all__ = ['FlowFilter']


class FlowFilter(object):
    """Superclass for Filter algorithms.

    It provides a basic interface to load images into the
    filter, compute optical flow and retrieve runtime.
    """

    def __init__(self):
        """Base constructor, empty implementation
        """

        pass


    def loadImage(self, img):
        """Load a new image to the filter

        Parameters
        ----------
        img : ndarray
            Image data
        """

        pass


    def compute(self):
        """Performs filter computations

        This method does not return the computed
        flow.

        See also
        --------
        getFlow : returns optical flow
        """

        pass


    def elapsedTime(self):
        """Returs the runtime taken to compute

        Returns
        -------
        runtime : float
            Elapsed time in milliseconds

        See also
        --------
        compute : Performs filter computations
        """

        pass


    def getFlow(self):
        """Returns optical flow
        """

        pass



class SimpleFlowFilter(FlowFilter):
    """Single pyramid level optical flow filter
    """

    def __init__(self, **kwargs):
        super(SimpleFlowFilter, self).__init__()

        # unroll kwargs
        self._propIterations = kwargs.pop('propIterations', 1)
        self._gamma = kwargs.pop('gamma', 1)
        self._smoothIterations = kwargs.pop('smoothIterations', 1)


        self._firstLoad = True
        self._elapsedTime = 0.0


    def loadImage(self, img):
        """Load a new image to the filter

        Parameters
        ----------
        img : ndarray
            Image data
        """

        self._img = img

        # if this is the firs loaded image
        # set _imgOld same as new image
        if self._firstLoad:
            self._imgOld = np.copy(img)

            # initializes flow
            self._flow = np.zeros((self._imgOld.shape[0], self._imgOld.shape[1], 2),
                dtype=np.float32)

            self._firstLoad = False


    def compute(self):
        """Performs filter computations

        This method does not return the computed
        flow.

        See also
        --------
        getFlow : returns optical flow
        """

        # start recording elapsed time
        start = time.clock()

        # propagation
        self._flow, _ = prop.propagate(self._flow, self._propIterations)

        # update
        self._flow, self._imgOld = upd.update(self._img, self._imgOld,
            self._flow, gamma=self._gamma)
        
        # truncate flow to safe range for propagation
        self._flow[self._flow > self._propIterations] = self._propIterations
        self._flow[self._flow < -self._propIterations] = -self._propIterations
        
        # smoothing
        self._flow = upd.smoothFlow(self._flow, self._smoothIterations)
        

        # stop recording elapsed time
        stop = time.clock()

        # elapsed time in milliseconds
        self._elapsedTime = (stop - start) * 1000.0


    def elapsedTime(self):
        """Returs the runtime taken to compute

        Returns
        -------
        runtime : float
            Elapsed time in milliseconds

        See also
        --------
        compute : Performs filter computations
        """

        return self._elapsedTime


    def getFlow(self):
        """Returns optical flow
        """

        return self._flow

    # TODO: add properties for gamma, propIterations and smoothIterations



class PyramidalFlowFilter(FlowFilter):
    """Pyramidal optical flow filter
    """

    def __init__(self, **kwargs):
        pass


    def loadImage(self, img):
        """Load a new image to the filter

        Parameters
        ----------
        img : ndarray
            Image data
        """

        pass


    def compute(self):
        """Performs filter computations

        This method does not return the computed
        flow.

        See also
        --------
        getFlow : returns optical flow
        """

        pass


    def elapsedTime(self):
        """Returs the runtime taken to compute

        Returns
        -------
        runtime : float
            Elapsed time in milliseconds

        See also
        --------
        compute : Performs filter computations
        """

        pass


    def getFlow(self):
        """Returns optical flow
        """

        pass