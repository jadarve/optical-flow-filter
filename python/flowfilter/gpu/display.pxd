"""
    flowfilter.gpu.display
    ----------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.gpu.image as gimg

cdef extern from 'flowfilter/gpu/display.h' namespace 'flowfilter::gpu':
    
    cdef cppclass FlowToColor_cpp 'flowfilter::gpu::FlowToColor':

        FlowToColor_cpp()
        FlowToColor_cpp(gimg.GPUImage_cpp inputFlow, const float maxflow)


        void configure()
        void compute()
        float elapsedTime()


        # Pipeline stage inputs
        void setInputFlow(gimg.GPUImage_cpp inputFlow)

        # Pipeline stage outputs
        gimg.GPUImage_cpp getColorFlow()

        # Parameters
        float getMaxFlow() const
        void setMaxFlow(const float maxflow)


cdef class FlowToColor:
    
    cdef FlowToColor_cpp flowColor