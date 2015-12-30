"""
    flowfilter.gpu.imagemodel
    -------------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.gpu.image as gimg

cdef extern from 'flowfilter/gpu/imagemodel.h' namespace 'flowfilter::gpu':
    
    cdef cppclass ImageModel_cpp 'flowfilter::gpu::ImageModel':

        ImageModel_cpp()
        ImageModel_cpp(gimg.GPUImage_cpp inputImage)


        void configure()
        void compute()
        float elapsedTime()

        # Pipeline stage inputs
        void setInputImage(gimg.GPUImage_cpp img)

        # Pipeline stage outputs
        gimg.GPUImage_cpp getImageConstant()
        gimg.GPUImage_cpp getImageGradient()


cdef class ImageModel:
    
    cdef ImageModel_cpp imodel

