"""
    flowfilter.gpu.propagation
    --------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

from libcpp cimport bool

cimport flowfilter.gpu.image as gimg

cdef extern from 'flowfilter/gpu/propagation.h' namespace 'flowfilter::gpu':
    
    cdef cppclass FlowPropagator_cpp 'flowfilter::gpu::FlowPropagator':

        FlowPropagator_cpp()
        FlowPropagator_cpp(gimg.GPUImage_cpp inputFlow,
            const int iterations)


        void configure()
        void compute()
        float elapsedTime()

        void setIterations(const int N)
        int getIterations() const
        float getDt() const

        void setBorder(const int border)
        int getBorder() const

        void setInvertInputFlow(const bool invert)
        bool getInvertInputFlow() const

        # Pipeline stage inputs
        void setInputFlow(gimg.GPUImage_cpp inputFlow)

        # Pipeline stage outputs
        gimg.GPUImage_cpp getPropagatedFlow()


    cdef cppclass FlowPropagatorPayload_cpp 'flowfilter::gpu::FlowPropagatorPayload':

        FlowPropagatorPayload_cpp()
        FlowPropagatorPayload_cpp(gimg.GPUImage_cpp inputFlow,
            gimg.GPUImage_cpp scalarPayload,
            gimg.GPUImage_cpp vectorPayload,
            const int iterations)

        void configure()
        void compute()
        float elapsedTime()

        void setIterations(const int N)
        int getIterations() const
        float getDt() const

        void setBorder(const int border)
        int getBorder() const

        # Pipeline stage inputs
        void setInputFlow(gimg.GPUImage_cpp img)
        void setScalarPayload(gimg.GPUImage_cpp img)
        void setVectorPayload(gimg.GPUImage_cpp img)

        # Pipeline stage outputs
        gimg.GPUImage_cpp getPropagatedFlow()
        gimg.GPUImage_cpp getPropagatedScalar()
        gimg.GPUImage_cpp getPropagatedVector()



    cdef cppclass LaxWendroffPropagator_cpp 'flowfilter::gpu::LaxWendroffPropagator':

        LaxWendroffPropagator_cpp()
        LaxWendroffPropagator_cpp(gimg.GPUImage_cpp inputFlow,
            gimg.GPUImage_cpp inputImage)

        void configure()
        void compute()
        float elapsedTime()

        void setIterations(const int N)
        int getIterations() const
        float getDt() const

        # Pipeline stage inputs
        void setInputFlow(gimg.GPUImage_cpp inputFlow);
        void setInputImage(gimg.GPUImage_cpp img);

        # Pipeline stage outputs
        gimg.GPUImage_cpp getFlow();
        gimg.GPUImage_cpp getPropagatedImage();


cdef class FlowPropagator:
    
    cdef FlowPropagator_cpp propagator


cdef class FlowPropagatorPayload:
    
    cdef FlowPropagatorPayload_cpp propagator


cdef class LaxWendroffPropagator:

    cdef LaxWendroffPropagator_cpp propagator