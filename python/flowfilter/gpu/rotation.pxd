"""
    flowfilter.gpu.rotation
    -----------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport flowfilter.gpu.image as gimg
cimport flowfilter.gpu.camera as gcam

cdef extern from 'flowfilter/gpu/rotation.h' namespace 'flowfilter::gpu':
    

    cdef cppclass RotationalFlowImagePredictor_cpp 'flowfilter::gpu::RotationalFlowImagePredictor':
    

        RotationalFlowImagePredictor_cpp()
        RotationalFlowImagePredictor_cpp(gcam.perspectiveCamera_cpp cam)
        RotationalFlowImagePredictor_cpp(gcam.perspectiveCamera_cpp cam,
            gimg.GPUImage_cpp inputImage)


        void configure()
        void compute()
        float elapsedTime()

        # Stage inputs
        void setInputImage(gimg.GPUImage_cpp inputImage)

        # Stage outputs
        gimg.GPUImage_cpp getPredictedImage()
        gimg.GPUImage_cpp getOpticalFlow()

        # Parameters
        void setCamera(gcam.perspectiveCamera_cpp cam)
        void setAngularVelocity(const float wx, const float wy, const float wz)
        void setIterations(const int iterations)
        int getIterations() const



cdef class RotationalFlowImagePredictor:
    
    cdef RotationalFlowImagePredictor_cpp predictor
