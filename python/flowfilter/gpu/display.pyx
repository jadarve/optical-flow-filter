"""
    flowfilter.gpu.display
    ----------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport numpy as np
import numpy as np

cimport flowfilter.image as fimg
import flowfilter.image as fimg

cimport flowfilter.gpu.image as gimg
import flowfilter.gpu.image as gimg

cdef class FlowToColor:
    
    def __cinit__(self, gimg.GPUImage inputFlow = None,
        float maxflow = 1.0):

        if inputFlow == None:
            return

        self.flowColor = FlowToColor_cpp(inputFlow.img, maxflow)


    def __dealloc__(self):
        # nothing to do
        pass

    def configure(self):
        self.flowColor.configure()


    def compute(self):
        self.flowColor.compute()


    def elapsedTime(self):
        return self.flowColor.elapsedTime()


    def setInputFlow(self, gimg.GPUImage inputFlow):
        """
        """

        self.flowColor.setInputFlow(inputFlow.img)


    def getColorFlow(self):

        cdef gimg.GPUImage colorFlow = gimg.GPUImage()
        colorFlow.img = self.flowColor.getColorFlow()

        return colorFlow


    def download(self):

        return self.getColorFlow().download(np.uint8)


    property maxflow:
        def __get__(self):
            return self.flowColor.getMaxFlow()

        def __set__(self, float value):
            self.flowColor.setMaxFlow(value)

        def __del__(self):
            pass