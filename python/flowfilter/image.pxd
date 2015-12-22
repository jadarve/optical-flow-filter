"""
    flowfilter.image
    ----------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cdef extern from 'flowfilter/image.h' namespace 'flowfilter':
    
    ctypedef struct image_t_cpp 'flowfilter::image_t':

        int height
        int width
        int depth
        size_t pitch
        size_t itemSize
        void* data



cdef class Image:
    
    cdef object numpyArray
    cdef image_t_cpp img
