"""
    flowfilter.gpu.pyramid
    ----------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.gpu.image as gimg

cdef extern from 'flowfilter/gpu/pyramid.h' namespace 'flowfilter::gpu':
    
    cdef cppclass ImagePyramid_cpp 'flowfilter::gpu::ImagePyramid':

        ImagePyramid_cpp()
        ImagePyramid_cpp(gimg.GPUImage_cpp image, const int levels)

        void configure()
        void compute()
        float elapsedTime()

        # Pipeline stage inputs
        void setInputImage(gimg.GPUImage_cpp img)
        void setLevels(const int levels)

        # Pipeline stage outputs
        gimg.GPUImage_cpp getImage(int level)
        int getLevels() const;
        

cdef class ImagePyramid:
    
    cdef ImagePyramid_cpp pyramid

