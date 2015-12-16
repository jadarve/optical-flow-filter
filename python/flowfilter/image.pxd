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
        size_t pitch
        unsigned char* ptr



cdef class Image:

    cdef image_t_cpp img
