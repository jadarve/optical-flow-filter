"""
    flowfilter.gpu.update
    --------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.gpu.image as gimg
import flowfilter.gpu.image as gimg


cdef class FlowUpdate:
    
    def __cinit__(self,
        gimg.GPUImage inputFlow = None,
        gimg.GPUImage inputImage = None,
        gimg.GPUImage inputImageGradient = None,
        float gamma = 1.0,
        float maxflow = 1.0):

        if inputFlow == None or inputImage == None or inputImageGradient == None:
            self.flowUpd.setGamma(gamma)
            self.flowUpd.setMaxFlow(maxflow)
            return

        self.flowUpd = FlowUpdate_cpp(inputFlow.img, inputImage.img,
            inputImageGradient.img, gamma, maxflow)


    def __dealloc__(self):
        # nothing to do
        pass

    def configure(self):
        self.flowUpd.configure()


    def compute(self):
        self.flowUpd.compute()


    def elapsedTime(self):
        return self.flowUpd.elapsedTime()


    def setInputFlow(self, gimg.GPUImage inputFlow):
        self.flowUpd.setInputFlow(inputFlow.img)


    def setInputImage(self, gimg.GPUImage image):
        self.flowUpd.setInputImage(image.img)


    def setInputImageGradient(self, gimg.GPUImage imageGradient):
        self.flowUpd.setInputImageGradient(imageGradient.img)


    def getUpdatedFlow(self):

        cdef gimg.GPUImage updFlow = gimg.GPUImage()
        updFlow.img = self.flowUpd.getUpdatedFlow()

        return updFlow


    def getUpdatedImage(self):

        cdef gimg.GPUImage updImg = gimg.GPUImage()
        updImg.img = self.flowUpd.getUpdatedImage()

        return updImg


    property gamma:

        def __get__(self):
            return self.flowUpd.getGamma()

        def __set__(self, float value):
            self.flowUpd.setGamma(value)

        def __del__(self):
            pass


    property maxflow:

        def __get__(self):
            return self.flowUpd.getMaxFlow()

        def __set__(self, float value):
            self.flowUpd.setMaxFlow(value)

        def __del__(self):
            pass



cdef class DeltaFlowUpdate:
    
    def __cinit__(self,
        gimg.GPUImage inputFlow = None,
        gimg.GPUImage inputDeltaFlow = None,
        gimg.GPUImage inputImageOld = None,
        gimg.GPUImage inputImage = None,
        gimg.GPUImage inputImageGradient = None,
        float gamma = 1.0,
        float maxflow = 1.0):

        if (inputFlow == None or inputDeltaFlow == None or
            inputImageOld == None or inputImage == None or
            inputImageGradient == None):

            self.deltaFlowUpd.setGamma(gamma)
            self.deltaFlowUpd.setMaxFlow(maxflow)
            return

        self.deltaFlowUpd = DeltaFlowUpdate_cpp(inputFlow.img,
            inputDeltaFlow.img, inputImageOld.img,
            inputImage.img, inputImageGradient.img,
            gamma, maxflow)


    def __dealloc__(self):
        # nothing to do
        pass

    def configure(self):
        self.deltaFlowUpd.configure()


    def compute(self):
        self.deltaFlowUpd.compute()


    def elapsedTime(self):
        return self.deltaFlowUpd.elapsedTime()


    def setInputFlow(self, gimg.GPUImage inputFlow):
        self.deltaFlowUpd.setInputFlow(inputFlow.img)


    def setInputDeltaFlow(self, gimg.GPUImage inputDeltaFlow):
        self.deltaFlowUpd.setInputDeltaFlow(inputDeltaFlow.img)


    def setInputImage(self, gimg.GPUImage image):
        self.deltaFlowUpd.setInputImage(image.img)


    def setInputImageGradient(self, gimg.GPUImage imageGradient):
        self.deltaFlowUpd.setInputImageGradient(imageGradient.img)


    def getUpdatedFlow(self):

        cdef gimg.GPUImage updFlow = gimg.GPUImage()
        updFlow.img = self.deltaFlowUpd.getUpdatedFlow()

        return updFlow


    def getUpdatedDeltaFlow(self):

        cdef gimg.GPUImage updFlow = gimg.GPUImage()
        updFlow.img = self.deltaFlowUpd.getUpdatedDeltaFlow()

        return updFlow


    def getUpdatedImage(self):

        cdef gimg.GPUImage updImg = gimg.GPUImage()
        updImg.img = self.deltaFlowUpd.getUpdatedImage()

        return updImg


    property gamma:

        def __get__(self):
            return self.deltaFlowUpd.getGamma()

        def __set__(self, float value):
            self.deltaFlowUpd.setGamma(value)

        def __del__(self):
            pass


    property maxflow:

        def __get__(self):
            return self.deltaFlowUpd.getMaxFlow()

        def __set__(self, float value):
            self.deltaFlowUpd.setMaxFlow(value)

        def __del__(self):
            pass