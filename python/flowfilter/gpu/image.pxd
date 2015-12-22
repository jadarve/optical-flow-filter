"""
    flowfilter.gpu.image
    --------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.image as fimg

cdef extern from 'flowfilter/gpu/image.h' namespace 'flowfilter::gpu':
    
    cdef cppclass GPUImage_cpp 'flowfilter::gpu::GPUImage':

        GPUImage_cpp();
        GPUImage_cpp(const int height, const int width,
            const int depth, const int itemSize);

        int height() const;
        int width() const;
        int depth() const;
        int pitch() const;
        int itemSize() const;

        void upload(fimg.image_t_cpp& img);
        void download(fimg.image_t_cpp& img) const;


cdef class GPUImage:
        
    cdef GPUImage_cpp img


