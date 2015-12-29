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


    def getFlow(self, np.ndarray flow = None):

        if flow == None:
            flow = np.zeros((self.height, self.width, 2), dtype=np.float32)

        # wrap numpy array in a Image object
        cdef fimg.Image flow_w = fimg.Image(flow)

        # transfer image to device memory space
        self.ffilter.downloadFlow(flow_w.img)

        return flow


    def getImage(self, np.ndarray image = None):

        if image == None:
            image = np.zeros((self.height, self.width), dtype=np.float32)

        # wrap numpy array in a Image object
        cdef fimg.Image image_w = fimg.Image(image)

        # transfer image to device memory space
        self.ffilter.downloadImage(image_w.img)

        return image


    #def getImageGradient(self, np.ndarray gradient = None):

    #    if gradient == None:
    #        gradient = np.zeros((self.height, self.width, 2), dtype=np.float32)

    #    # wrap numpy array in a Image object
    #    cdef fimg.Image gradient_w = fimg.Image(gradient)

    #    # transfer image to device memory space
    #    self.ffilter.downloadImageGradient(gradient_w.img)

    #    return gradient


    #def getImageConstant(self, np.ndarray image = None):

    #    if image == None:
    #        image = np.zeros((self.height, self.width), dtype=np.float32)

    #    # wrap numpy array in a Image object
    #    cdef fimg.Image image_w = fimg.Image(image)

    #    # transfer image to device memory space
    #    self.ffilter.downloadImageConstant(image_w.img)

    #    return image


    #def getImageUpdated(self, np.ndarray image = None):

    #    if image == None:
    #        image = np.zeros((self.height, self.width), dtype=np.float32)

    #    # wrap numpy array in a Image object
    #    cdef fimg.Image image_w = fimg.Image(image)

    #    # transfer image to device memory space
    #    self.ffilter.downloadImageUpdated(image_w.img)

    #    return image


    #def getSmoothedFlow(self, np.ndarray flow = None):

    #    if flow == None:
    #        flow = np.zeros((self.height, self.width, 2), dtype=np.float32)

    #    # wrap numpy array in a Image object
    #    cdef fimg.Image flow_w = fimg.Image(flow)

    #    # transfer image to device memory space
    #    self.ffilter.downloadSmoothedFlow(flow_w.img)

    #    return flow


    def configure(self):
        self.ffilter.configure()


    def compute(self):
        self.ffilter.compute()


    def elapsedTime(self):
        return self.ffilter.elapsedTime()


    #############################################
    # PROPERTIES
    #############################################

    property height:
        def __get__(self):
            return self.ffilter.height()

        def __set__(self, value):
            raise RuntimeError('Filter height cannot be set after instantiation')

        def __del__(self):
            pass


    property width:
        def __get__(self):
            return self.ffilter.width()

        def __set__(self, value):
            raise RuntimeError('Filter width cannot be set after instantiation')

        def __del__(self):
            pass


    property gamma:
        def __get__(self):
            return self.ffilter.getGamma()

        def __set__(self, float value):
            self.ffilter.setGamma(value)

        def __del__(self):
            pass


    property maxflow:
        def __get__(self):
            return self.ffilter.getMaxFlow()

        def __set__(self, float value):
            self.ffilter.setMaxFlow(value)

        def __del__(self):
            pass


    property smoothIterations:
        def __get__(self):
            return self.ffilter.getSmoothIterations()

        def __set__(self, int value):
            self.ffilter.setSmoothIterations(value)

        def __del__(self):
            pass


    property propagationIterations:
        def __get__(self):
            return self.ffilter.getPropagationIterations()

        def __set__(self, value):
            raise RuntimeError('propagation iterations cannot be set, use maxflow instead')

        def __del__(self):
            pass
