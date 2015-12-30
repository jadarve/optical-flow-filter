"""
    flowfilter.gpu.imagemodel
    -------------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""


cimport flowfilter.gpu.image as gimg
import flowfilter.gpu.image as gimg


cdef class ImageModel:
    
    def __cinit__(self, gimg.GPUImage inputImage = None):
        
        if inputImage == None:
            # nothing to do
            return
        
        self.imodel = ImageModel_cpp(inputImage.img)


    def __dealloc__(self):
        # nothing to do
        pass

    def configure(self):
        self.imodel.configure()


    def compute(self):
        self.imodel.compute()


    def elapsedTime(self):
        return self.imodel.elapsedTime()


    def setInputImage(self, gimg.GPUImage inputImage):
        """
        """

        self.imodel.setInputImage(inputImage.img)


    def getImageConstant(self):
        """
        """

        cdef gimg.GPUImage img = gimg.GPUImage()
        img.img = self.imodel.getImageConstant()

        return img

    def getImageGradient(self):
        """
        """

        cdef gimg.GPUImage imgGrad = gimg.GPUImage()
        imgGrad.img = self.imodel.getImageGradient()

        return imgGrad