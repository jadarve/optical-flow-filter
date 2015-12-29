"""
    flowfilter.gpu.flowfilter
    -------------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.gpu.image as gimg
cimport flowfilter.image as fimg

cdef extern from 'flowfilter/gpu/flowfilter.h' namespace 'flowfilter::gpu':
    
    cdef cppclass FlowFilter_cpp 'flowfilter::gpu::FlowFilter':

        FlowFilter_cpp()
        FlowFilter_cpp(const int height, const int witdh)
        FlowFilter_cpp(const int height, const int witdh,
                       const int smoothIterations,
                       const float maxflow,
                       const float gamma)


        void configure()
        void compute()
        float elapsedTime()

        # Pipeline stage output
        gimg.GPUImage_cpp getFlow()
       
        # Host load-download
        void loadImage(fimg.image_t_cpp& image)
        void downloadFlow(fimg.image_t_cpp& flow)
        void downloadImage(fimg.image_t_cpp& image);

        ## Image model outputs
        #void downloadImageGradient(fimg.image_t_cpp& gradient)
        #void downloadImageConstant(fimg.image_t_cpp& gradient)

        ## Update stage outputs
        #void downloadImageUpdated(fimg.image_t_cpp& image)

        ## Smooth stage outputs
        #void downloadSmoothedFlow(fimg.image_t_cpp& flow)
        
        # Parameters
        float getGamma() const
        void setGamma(const float gamma)

        float getMaxFlow() const
        void setMaxFlow(const float maxflow)

        int getSmoothIterations() const
        void setSmoothIterations(const int N)

        int getPropagationIterations() const

        int height() const
        int width() const


cdef class FlowFilter:
    
    cdef FlowFilter_cpp ffilter

