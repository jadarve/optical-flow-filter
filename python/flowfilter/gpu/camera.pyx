"""
    flowfilter.gpu.camera
    ---------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

import numpy as np

cdef class PerspectiveCamera:
    

    def __init__(self):
        """Perspective camera class wrapper
        """
        pass


    def __cinit__(self):
        pass


    def __dealloc__(self):
        pass


    def getIntrinsicsMatrix(self):

        K = np.zeros((3,3), dtype=np.float32)
        K[0,0] = self.alphaX
        K[0,2] = self.centerX
        K[1,1] = self.alphaY
        K[1,2] = self.centerY
        K[2,2] = 1.0

        return K


    property alphaX:
        def __get__(self):
            return self.cam.alphaX

        def __set__(self, value):
            raise RuntimeError('alphaX cannot be set')

        def __del__(self):
            pass


    property alphaY:
        def __get__(self):
            return self.cam.alphaY

        def __set__(self, value):
            raise RuntimeError('alphaY cannot be set')

        def __del__(self):
            pass


    property centerX:
        def __get__(self):
            return self.cam.centerX

        def __set__(self, value):
            raise RuntimeError('centerX cannot be set')

        def __del__(self):
            pass


    property centerY:
        def __get__(self):
            return self.cam.centerY

        def __set__(self, value):
            raise RuntimeError('centerY cannot be set')

        def __del__(self):
            pass


def createPerspectiveCamera(const float focalLength, const int height, const int width,
        const float sensorHeight, const float sensorWidth):
    

    cdef PerspectiveCamera cam = PerspectiveCamera()
    cam.cam = createPerspectiveCamera_cpp(focalLength, height, width,
        sensorHeight, sensorWidth)

    return cam
