"""
    flowfilter.gpu.pyramid
    ----------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.gpu.image as gimg
import flowfilter.gpu.image as gimg


cdef class ImagePyramid:
    
    def __cinit__(self, gimg.GPUImage inputImage = None, int levels=1):
        
        if inputImage == None:
            # nothing to do
            return
        
        self.pyramid = ImagePyramid_cpp(inputImage.img, levels)


    def __dealloc__(self):
        # nothing to do
        pass

    def configure(self):
        self.pyramid.configure()


    def compute(self):
        self.pyramid.compute()


    def elapsedTime(self):
        return self.pyramid.elapsedTime()


    def setInputImage(self, gimg.GPUImage inputImage):
        """
        """

        self.pyramid.setInputImage(inputImage.img)


    def getImage(self, int level):
        """
        """

        cdef gimg.GPUImage img = gimg.GPUImage()
        img.img = self.pyramid.getImage(level)

        return img


    property levels:
        def __get__(self):
            return self.pyramid.getLevels()

        def __set__(self, int levels):
            self.pyramid.setLevels(levels)

        def __del__(self):
            pass