"""
    flowfilter.gpu.rotation
    -----------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""


cimport flowfilter.gpu.image as gimg
import flowfilter.gpu.image as gimg


cimport flowfilter.gpu.camera as gcam
import flowfilter.gpu.camera as gcam


cdef class RotationalFlowImagePredictor:
    

    def __init__(self, gcam.PerspectiveCamera cam,
        gimg.GPUImage inputImage=None):
        """Rotational flow image predictor

        Parameters
        ----------
        cam : PerspectiveCamera, optional.
            Camera model for the acquired images. Defaults to None.

        inputImage : GPUImage, optional.
            Input image buffer. Defaults to None.
        """
        pass


    def __cinit__(self, gcam.PerspectiveCamera cam,
        gimg.GPUImage inputImage=None):
        
        if cam is None or inputImage is None:
            return

        self.predictor = RotationalFlowImagePredictor_cpp(cam.cam, inputImage.img)


    def configure(self):
        self.predictor.configure()


    def compute(self):
        self.predictor.compute()


    def elapsedTime(self):
        return self.elapsedTime()


    def setInputImage(self, gimg.GPUImage inputImage):
        
        self.predictor.setInputImage(inputImage.img)


    def getPredictedImage(self):

        cdef gimg.GPUImage img = gimg.GPUImage()
        img.img = self.predictor.getPredictedImage()

        return img


    def getOpticalFlow(self):
        
        cdef gimg.GPUImage flow = gimg.GPUImage()
        flow.img = self.predictor.getOpticalFlow()

        return flow


    def setCamera(self, gcam.PerspectiveCamera cam):
        
        self.predictor.setCamera(cam.cam)


    def setAngularVelocity(self, float wx, float wy, float wz):
        
        self.predictor.setAngularVelocity(wx, wy, wz);


    property iterations:
        def __get__(self):
            return self.predictor.getIterations()

        def __set__(self, int N):
            self.predictor.setIterations(N)

        def __del__(self):
            pass

