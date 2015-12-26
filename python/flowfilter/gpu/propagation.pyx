"""
    flowfilter.gpu.propagation
    --------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.gpu.image as gimg
import flowfilter.gpu.image as gimg


cdef class FlowPropagator:
    
    def __cinit__(self, gimg.GPUImage inputFlow = None,
        int iterations = 1):
        
        if inputFlow == None:
            # nothing to do
            return
        
        self.propagator = FlowPropagator_cpp(inputFlow.img, iterations)


    def __dealloc__(self):
        # nothing to do
        pass

    def configure(self):
        self.propagator.configure()


    def compute(self):
        self.propagator.compute()


    def elapsedTime(self):
        return self.propagator.elapsedTime()


    def setInputFlow(self, gimg.GPUImage inputFlow):
        """
        """

        self.propagator.setInputFlow(inputFlow.img)


    def getPropagatedFlow(self):

        cdef gimg.GPUImage propFlow = gimg.GPUImage()
        propFlow.img = self.propagator.getPropagatedFlow()

        return propFlow
    
    