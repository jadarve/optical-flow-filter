"""
    flowfilter.gpu.propagation
    --------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.gpu.image as gimg

cdef extern from 'flowfilter/gpu/propagation.h' namespace 'flowfilter::gpu':
    
    cdef cppclass FlowPropagator_cpp 'flowfilter::gpu::FlowPropagator':

        FlowPropagator_cpp()
        FlowPropagator_cpp(gimg.GPUImage_cpp inputFlow, const int iterations)


        void configure()
        void compute()
        float elapsedTime()

        void setIterations(const int N)
        int getIterations() const
        float getDt() const

        # Pipeline stage inputs
        void setInputFlow(gimg.GPUImage_cpp inputFlow)

        # Pipeline stage outputs
        gimg.GPUImage_cpp getPropagatedFlow()



cdef class FlowPropagator:
    
    cdef FlowPropagator_cpp propagator

