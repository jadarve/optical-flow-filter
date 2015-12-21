"""
    flowfilter.flowfilter
    ---------------------

    Module containing abstract filter classes and Python
    implementation of the optical flow filter algorithm

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import math
import time
from collections import Iterable

import numpy as np

from . import propagation as prop
from . import update as upd
from . import misc as fmisc


__all__ = ['FlowFilter', 'SimpleFlowFilter',
    'DeltaFlowFilter', 'PyramidalFlowFilter']


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

    Examples
    --------
    """

    def __init__(self, **kwargs):
        """Creates a new filter instance

        Kwargs
        ------
        propIterations : integer, optional
            Number of iterations performed during the
            propagation stage. This parameter also controls
            the maximum flow value the filter can handle. If
            it is set to N, then the filter can handle up to
            N pixels/frame optical flow values on each component.
            Defaults to 1.

        smoothIterations : integer, optional
            Number of smooth iterations applied after the update
            stage of the filter. Defaults to 1.

        gamma : float, optional
            temporal regularization gain controlling the relevance
            of the predicted flow in the update. Value should be
            greater than 0.0. Defaults to 1.0.

        """

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

        First it propagates old estimation of flow to current
        time. Second, it updates the propagated flow with the
        new uploaded image. Finally, it applies a smooting
        operator to the updated flow to spread information to
        textureless regions of the image.

        See also
        --------
        loadImage : load a new image to the filter
        getFlow : returns optical flow
        elapsedTime : returns the runtime taken to compute flow
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
        """Returns the runtime taken to compute flow

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



class DeltaFlowFilter(FlowFilter):

    def __init__(self, **kwargs):
        """Creates a new DeltaFlowFilter instance

        Kwargs
        ----------
        propIterations : integer, optional
            Propagation iterations. Defaults to 1

        smoothIterations : integer, optional
            Smooth iterations applied to delta flow after update. Defaults to 1

        gamma : float, optional
            Temporal regularization gain. This term controls the weight
            of the propagated flow in the update stage of the filter.
            Defaults to 1.0

        maxflow : float, optional
            Max delta flow value allowed for during the update stage of the filter.
            Defaults to 0.25.
        """

        super(DeltaFlowFilter, self).__init__()

        # unroll kwargs
        self._propIterations = kwargs.pop('propIterations', 1)
        self._gamma = kwargs.pop('gamma', 1.0)
        self._smoothIterations = kwargs.pop('smoothIterations', 1)
        self._maxflow = kwargs.pop('maxflow', 0.25)

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

            # initializes delta flow
            self._deltaFlow = np.zeros((self._imgOld.shape[0], self._imgOld.shape[1], 2),
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

        ###############################
        # propagation
        ###############################

        # payloads correspond to delta flow and old image
        payload = [np.copy(self._deltaFlow[...,0]), np.copy(self._deltaFlow[...,1]), self._imgOld]

        # truncate flow to safe range for propagation
        self._flow[self._flow > self._propIterations] = self._propIterations
        self._flow[self._flow < -self._propIterations] = -self._propIterations

        # propagate with payload
        self._flow, payload = prop.propagate(self._flow, self._propIterations, payload=payload)

        # reconstruct deltaFlow and imgOld from propagated payloads
        self._deltaFlow = np.concatenate([p[...,np.newaxis] for p in payload[0:2]], axis=2)
        self._imgOld = payload[2]

        ###############################
        # update
        ###############################
        self._deltaFlow, self._imgOld = upd.update(self._img, self._imgOld,
            self._deltaFlow, gamma=self._gamma)
        
        # truncate deltaFlow to [-maxflow, maxflow]
        self._deltaFlow[self._deltaFlow > self._maxflow] = self._maxflow
        self._deltaFlow[self._deltaFlow < -self._maxflow] = -self._maxflow

        # smoothing
        self._deltaFlow = upd.smoothFlow(self._deltaFlow, self._smoothIterations)

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


    def setFlow(self, flow):
        """Set optical flow

        Parameters
        ----------
        flow : ndarray
            Optical flow field.
        """

        self._flow = flow


    def getDeltaFlow(self):
        """Returns delta optical flow
        """

        return self._deltaFlow


class PyramidalFlowFilter(FlowFilter):
    """Pyramidal optical flow filter

    See also
    --------
    SimpleFlowFilter :
    DeltaFlowFilter : 

    Examples
    --------
    """

    def __init__(self, **kwargs):
        """Creates a new instance of PyramidalFlowFilter

        Kwargs
        ----------
        levels : integer, optional
            Pyramid levels. Defaults to 1.

        propIterations : integer or list[integer], optional
            Propagation iterations. If integer, then the iterations for the
            base level are set to this value. Higher levels of the pyramid
            are set to ceil(N / 2^h). If the parameter is a list, then the iterations
            for each level are set to the value found in the list.
            Defaults to 1.

        smoothIterations : integer or list[integer], optional
            Smooth iterations applied to delta flow after update stage at
            each level of the pyramid. Defaults to 1

        gamma : float or list[float], optional
            Temporal regularization gain for each level. This term controls the weight
            of the propagated flow in the update stage of the filter.
            Defaults to 1.0

        maxflow : float, optional
            Max delta flow value allowed for during the update stage at low levels
            of the pyramid. Defaults to 0.25.
        """
        
        # unroll kwargs
        self._H = kwargs.pop('levels', 1)               # pyramid levels
        pIter = kwargs.pop('propIterations', 1)         # propagation iterations
        sIter = kwargs.pop('smoothIterations', 1)       # smooth iterations
        gamma = kwargs.pop('gamma', 1.0)                # gains
        self._maxflow = kwargs.pop('maxflow', 0.25)     # maxflow for delta flow filters

        # creates parameters for each level
        self._propIterations = list()
        self._smoothIterations = list()
        self._gamma = list()

        # create parameter lists for all levels
        for h in range(self._H):

            if isinstance(pIter, Iterable):
                self._propIterations.append(pIter[h])
            else:
                self._propIterations.append(int(math.ceil(pIter / math.pow(2, h))))
            

            self._smoothIterations.append(sIter[h] if isinstance(sIter, Iterable) else sIter)
            self._gamma.append(gamma[h] if isinstance(gamma, Iterable) else gamma)


        # create filter blocks at each level
        # top level
        self._filterTop = SimpleFlowFilter(propIterations=self._propIterations[self._H-1],
            smoothIterations=self._smoothIterations[self._H-1], gamma=self._gamma[self._H-1])

        # low levels
        self._lowLevelFilters = list()
        for h in range(self._H-1):
            filterLow = DeltaFlowFilter(proptIterations=self._propIterations[h],
                smoothIterations=self._smoothIterations[h], gamma=self._gamma[h], maxflow=self._maxflow)

            self._lowLevelFilters.append(filterLow)

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

        # if this is the firs loaded image, initialize flow
        if self._firstLoad:

            # initializes flow
            self._flow = np.zeros((self._img.shape[0], self._img.shape[1], 2),
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

        # compute pyramid for input image
        imgPyr = fmisc.imagePyramid(self._img, self._H)

        # load image at top level
        self._filterTop.loadImage(imgPyr[-1])

        # top level optical flow before update
        flowOld = self._filterTop.getFlow()

        # compute top level optical flow
        self._filterTop.compute()

        # new estimate of optical flow
        self._flow = self._filterTop.getFlow()

        # if there are more levels in the pyramid
        if self._H > 1:

            # lower levels computation
            for h in range(self._H-2, -1, -1):

                filterLow = self._lowLevelFilters[h]

                # load image
                filterLow.loadImage(imgPyr[h])

                # print('Delta flow shape: {0}'.format(filterLow.getDeltaFlow().shape))
                # print('Flow old shape: {0}'.format(flowOld.shape))

                # upsample upper level flow and add old estimate Delta flow
                flowOld = 2.0*fmisc.imageUp(flowOld) + filterLow.getDeltaFlow()

                # set old optical flow used for the propagation
                filterLow.setFlow(flowOld)

                # compute new Delta flow
                filterLow.compute()

                # upsample new estimated flow
                self._flow = 2.0*fmisc.imageUp(self._flow) + filterLow.getDeltaFlow()

                # truncate flow to safe range for propagation
                self._flow[self._flow > self._propIterations[h]] = self._propIterations[h]
                self._flow[self._flow < -self._propIterations[h]] = -self._propIterations[h]


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

