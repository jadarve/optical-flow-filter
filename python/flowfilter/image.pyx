"""
    flowfilter.image
    ----------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

from image cimport image_t_cpp


cdef class Image:
    """Image wrapper class"""

    def __cinit__(self):
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


    property pitch:
        def __get__(self):
            return self.img.pitch

        def __set__(self, v):
            raise RuntimeError('Image pitch cannot be set')

        def __del__(self):
            pass    # nothing to do
