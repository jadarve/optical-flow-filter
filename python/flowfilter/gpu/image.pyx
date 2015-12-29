"""
    flowfilter.gpu.image
    --------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport numpy as np
import numpy as np

cimport flowfilter.image as fimg
import flowfilter.image as fimg


cdef class GPUImage:
    

    def __cinit__(self, shape = None, itemSize = 4):
        
        # validate shape
        if shape != None:
            ndim = len(shape)

            if ndim not in [2, 3]:
                raise ValueError('number of dimensions must be 2 or 3, got: {0}'.format(len(self.__shape)))

            # allocate device memory
            self.img = GPUImage_cpp(shape[0], shape[1],
                1 if ndim == 2 else shape[2], itemSize)

    def __dealloc__(self):
        # nothing to do
        pass

    def upload(self, np.ndarray img):
        
        # wrap numpy array in a Image object
        cdef fimg.Image img_w = fimg.Image(img)

        # transfer image to device memory space
        self.img.upload(img_w.img)


    def download(self, dtype, np.ndarray output=None):
        """Download image to numpy array

        Parameters
        ----------
        dtype : numpy dtype
            Numpy dtype of the downloaded image

        output : ndarray, optional
            Output numpy ndarray
        """

        if output == None:
            output = np.zeros(self.shape, dtype=dtype)

        oshape = (output.shape[0], output.shape[1], output.shape[2])

        cdef fimg.Image output_w = fimg.Image(output)
        self.img.download(output_w.img)
        
        return output


    property shape:

        def __get__(self):

            if self.img.depth() == 1:
                return (self.img.height(), self.img.width())
            else:
                return (self.img.height(), self.img.width(), self.img.depth())

        def __set__(self, value):
            raise RuntimeError('shape cannot be set')

        def __del__(self):
            pass # nothing to do


    property height:
        def __get__(self):
            return self.img.height()

        def __set__(self, value):
            raise RuntimeError('height cannot be set')

        def __del__(self):
            pass # nothing to do


    property width:
        def __get__(self):
            return self.img.width()

        def __set__(self, value):
            raise RuntimeError('width cannot be set')

        def __del__(self):
            pass # nothing to do


    property depth:
        def __get__(self):
            return self.img.depth()

        def __set__(self, value):
            raise RuntimeError('depth cannot be set')

        def __del__(self):
            pass # nothing to do
    

    property pitch:
        def __get__(self):
            return self.img.pitch()

        def __set__(self, value):
            raise RuntimeError('pitch cannot be set')

        def __del__(self):
            pass # nothing to do


    property itemSize:
        def __get__(self):
            return self.img.itemSize()

        def __set__(self, value):
            raise RuntimeError('itemSize cannot be set')

        def __del__(self):
            pass # nothing to do
