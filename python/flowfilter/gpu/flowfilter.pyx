"""
    flowfilter.gpu.flowfilter
    -------------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport numpy as np
import numpy as np

cimport flowfilter.image as fimg
import flowfilter.image as fimg

cdef class FlowFilter:

    
    def __cinit__(self, int height, int width,
        int smoothIterations = 1, float maxflow = 1.0,
        float gamma = 1.0):

        self.ffilter = FlowFilter_cpp(
            height, width, smoothIterations,
            maxflow, gamma)


    def loadImage(self, np.ndarray img):

        # wrap numpy array in a Image object
        cdef fimg.Image img_w = fimg.Image(img)

        # transfer image to device memory space
        self.ffilter.loadImage(img_w.img)


    def compute(self):
        self.ffilter.compute()


    def elapsedTime(self):
        return self.ffilter.elapsedTime()


    #############################################
    # PROPERTIES
    #############################################

