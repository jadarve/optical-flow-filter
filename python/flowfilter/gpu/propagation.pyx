"""
    flowfilter.gpu.propagation
    --------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

from cpython cimport bool

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


    property iterations:
        def __get__(self):
            return self.propagator.getIterations()

        def __set__(self, int N):
            self.propagator.setIterations(N)

        def __del__(self):
            pass
    

    property dt:
        def __get__(self):
            return self.propagator.getDt()

        def __set__(self, int dt):
            raise RuntimeError('dt cannot be set, use iterations property instead')

        def __del__(self):
            pass


    property invertInputFlow:
        def __get__(self):
            return self.propagator.getInvertInputFlow()

        def __set__(self, invert):
            self.propagator.setInvertInputFlow(invert)

        def __del__(self):
            pass


    property border:
        def __get__(self):
            return self.propagator.getBorder()

        def __set__(self, border):
            self.propagator.setBorder(border)

        def __del__(self):
            pass


cdef class FlowPropagatorPayload:
    
    def __cinit__(self,
        gimg.GPUImage inputFlow = None,
        gimg.GPUImage inputScalar = None,
        gimg.GPUImage inputVector = None,
        int iterations = 1):
        
        if (inputFlow == None or inputScalar == None
            or inputVector == None):

            # nothing to do
            return
        
        self.propagator = FlowPropagatorPayload_cpp(
            inputFlow.img, inputScalar.img,
            inputVector.img, iterations)


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


    def setScalarPayload(self, gimg.GPUImage scalarField):

        self.propagator.setScalarPayload(scalarField.img)


    def setVectorPayload(self, gimg.GPUImage vectorField):

        self.propagator.setVectorPayload(vectorField.img)


    def getPropagatedFlow(self):

        cdef gimg.GPUImage propFlow = gimg.GPUImage()
        propFlow.img = self.propagator.getPropagatedFlow()

        return propFlow


    def getPropagatedScalar(self):

        cdef gimg.GPUImage propScalar = gimg.GPUImage()
        propScalar.img = self.propagator.getPropagatedScalar()

        return propScalar


    def getPropagatedVector(self):

        cdef gimg.GPUImage propVector = gimg.GPUImage()
        propVector.img = self.propagator.getPropagatedVector()

        return propVector


    property iterations:
        def __get__(self):
            return self.propagator.getIterations()

        def __set__(self, int N):
            self.propagator.setIterations(N)

        def __del__(self):
            pass
    

    property dt:
        def __get__(self):
            return self.propagator.getDt()

        def __set__(self, int dt):
            raise RuntimeError('dt cannot be set, use iterations property instead')

        def __del__(self):
            pass


    property border:
        def __get__(self):
            return self.propagator.getBorder()

        def __set__(self, border):
            self.propagator.setBorder(border)

        def __del__(self):
            pass
