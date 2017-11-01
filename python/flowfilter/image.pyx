"""
    flowfilter.image
    ----------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport numpy as np
import numpy as np

from image cimport image_t_cpp


cdef class Image:
    """Image wrapper class"""

    def __cinit__(self, np.ndarray arr = None):

        if arr is None:
            self.numpyArray = None
            return

        if not arr.flags['C_CONTIGUOUS']:
            raise ValueError('arr must be C_CONTIGUOUS')

        # hold a reference to this numpy array inside this object
        self.numpyArray = arr

        # validate shape
        shape = arr.shape
        ndim = arr.ndim
        if ndim != 2 and ndim != 3:
            raise ValueError('Incorrect number of image dimensions. Expecting 2 or 3: {0}'.format(ndim))

        # populate image_t properties
        self.img.height = shape[0]
        self.img.width = shape[1]
        self.img.depth = shape[2] if ndim == 3 else 1
        self.img.pitch = arr.strides[0]              # first stride corresponds to row pitch
        self.img.itemSize = arr.strides[ndim -1]     # last stride corresponds to item size
        self.img.data = <void*>arr.data


    def __dealloc__(self):

        # nothing to do, memory is relased by self.numpyArray
        pass


    property width:
        def __get__(self):
            return self.img.width

        def __set__(self, v):
            raise RuntimeError('Image width cannot be set')

        def __del__(self):
            pass    # nothing to do


    property height:
        def __get__(self):
            return self.img.height

        def __set__(self, v):
            raise RuntimeError('Image height cannot be set')

        def __del__(self):
            pass    # nothing to do


    property depth:
        def __get__(self):
            return self.img.depth

        def __set__(self, v):
            raise RuntimeError('Image depth cannot be set')

        def __del__(self):
            pass    # nothing to do


    property itemSize:
        def __get__(self):
            return self.img.itemSize

        def __set__(self, v):
            raise RuntimeError('Image itemSize cannot be set')

        def __del__(self):
            pass    # nothing to do


    property pitch:
        def __get__(self):
            return self.img.pitch

        def __set__(self, v):
            raise RuntimeError('Image pitch cannot be set')

        def __del__(self):
            pass    # nothing to do
