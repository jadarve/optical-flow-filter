/**
 * \file rotation.h
 * \brief Classes for working with rotational optical flow fields.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#ifndef FLOWFILTER_GPU_ROTATION_H_
#define FLOWFILTER_GPU_ROTATION_H_


#include "flowfilter/osconfig.h"

#include "flowfilter/gpu/camera.h"
#include "flowfilter/gpu/image.h"
#include "flowfilter/gpu/pipeline.h"
#include "flowfilter/gpu/propagation.h"

namespace flowfilter {
namespace gpu {


class FLOWFILTER_API RotationalFlowImagePredictor : public Stage {


public:
    RotationalFlowImagePredictor();
    RotationalFlowImagePredictor(perspectiveCamera cam);
    RotationalFlowImagePredictor(perspectiveCamera cam,
        flowfilter::gpu::GPUImage inputImage);
    ~RotationalFlowImagePredictor();

public:
    /**
     * \brief configures the stage.
     *
     * After configuration, calls to compute()
     * are valid.
     * Input buffers should not change after
     * this method has been called.
     */
    void configure();

    /**
     * \brief perform computation
     */
    void compute();



    //#########################
    // Stage inputs
    //#########################
    void setInputImage(flowfilter::gpu::GPUImage inputImage);


    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getPredictedImage();
    flowfilter::gpu::GPUImage getOpticalFlow();


    //#########################
    // Parameters
    //#########################
    void setCamera(perspectiveCamera cam);
    void setAngularVelocity(const float wx, const float wy, const float wz);
    void setIterations(const int iterations);
    int getIterations() const;


private:
    bool __configured;
    bool __inputImageSet;

    perspectiveCamera __camera;
    float3 __angularVelocity;

    flowfilter::gpu::GPUImage __inputImage;
    flowfilter::gpu::GPUImage __opticalFlow;

    flowfilter::gpu::LaxWendroffPropagator __propagator;

    dim3 __grid;
    dim3 __block;
};


} // namespace gpu
} // namespace flowfilter

#endif /* FLOWFILTER_GPU_ROTATION_H_ */
