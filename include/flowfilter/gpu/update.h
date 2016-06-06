/**
 * \file update.h
 * \brief Optical flow filter update classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_UPDATE_H_
#define FLOWFILTER_GPU_UPDATE_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/osconfig.h"
#include "flowfilter/gpu/pipeline.h"
#include "flowfilter/gpu/image.h"

namespace flowfilter {
namespace gpu {

class FLOWFILTER_API FlowUpdate : public Stage {


public:
    FlowUpdate();
    FlowUpdate(flowfilter::gpu::GPUImage inputFlow,
               flowfilter::gpu::GPUImage inputImage,
               flowfilter::gpu::GPUImage inputImageGradient,
               const float gamma = 1.0,
               const float maxflow = 1.0);
    ~FlowUpdate();

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
     * \brief performs computation of brightness parameters
     */
    void compute();

    float getGamma() const;
    void setGamma(const float gamma);

    float getMaxFlow() const;
    void setMaxFlow(const float maxflow);

    //#########################
    // Stage inputs
    //#########################
    void setInputFlow(flowfilter::gpu::GPUImage inputFlow);
    void setInputImage(flowfilter::gpu::GPUImage image);
    void setInputImageGradient(flowfilter::gpu::GPUImage imageGradient);

    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getUpdatedFlow();
    flowfilter::gpu::GPUImage getUpdatedImage();


private:
    float __gamma;
    float __maxflow;

    bool __configured;
    bool __inputFlowSet;
    bool __inputImageSet;
    bool __inputImageGradientSet;

    flowfilter::gpu::GPUImage __inputFlow;
    flowfilter::gpu::GPUImage __inputImage;
    flowfilter::gpu::GPUImage __inputImageGradient;

    flowfilter::gpu::GPUImage __flowUpdated;
    flowfilter::gpu::GPUImage __imageUpdated;


    dim3 __block;
    dim3 __grid;

};


class FLOWFILTER_API DeltaFlowUpdate : public Stage {

public:
    DeltaFlowUpdate();
    DeltaFlowUpdate(flowfilter::gpu::GPUImage inputFlow,
                    flowfilter::gpu::GPUImage inputDeltaFlow,
                    flowfilter::gpu::GPUImage inputOldImage,
                    flowfilter::gpu::GPUImage inputImage,
                    flowfilter::gpu::GPUImage inputImageGradient,
                    const float gamma = 1.0,
                    const float maxflow = 1.0);
    ~DeltaFlowUpdate();

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
     * \brief performs computation of brightness parameters
     */
    void compute();

    float getGamma() const;
    void setGamma(const float gamma);

    float getMaxFlow() const;
    void setMaxFlow(const float maxflow);

    //#########################
    // Stage inputs
    //#########################
    void setInputFlow(flowfilter::gpu::GPUImage inputFlow);
    void setInputDeltaFlow(flowfilter::gpu::GPUImage inputDeltaFlow);
    void setInputImageOld(flowfilter::gpu::GPUImage image);
    void setInputImage(flowfilter::gpu::GPUImage image);
    void setInputImageGradient(flowfilter::gpu::GPUImage imageGradient);

    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getUpdatedFlow();
    flowfilter::gpu::GPUImage getUpdatedDeltaFlow();
    flowfilter::gpu::GPUImage getUpdatedImage();


private:
    float __gamma;
    float __maxflow;

    bool __configured;
    bool __inputDeltaFlowSet;
    bool __inputImageOldSet;
    bool __inputFlowSet;
    bool __inputImageSet;
    bool __inputImageGradientSet;

    flowfilter::gpu::GPUImage __inputFlow;
    flowfilter::gpu::GPUTexture __inputFlowTexture;

    flowfilter::gpu::GPUImage __inputDeltaFlow;
    flowfilter::gpu::GPUImage __inputImageOld;
    flowfilter::gpu::GPUImage __inputImage;
    flowfilter::gpu::GPUImage __inputImageGradient;

    flowfilter::gpu::GPUImage __flowUpdated;
    flowfilter::gpu::GPUImage __deltaFlowUpdated;
    flowfilter::gpu::GPUImage __imageUpdated;


    dim3 __block;
    dim3 __grid;

};

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_UPDATE_H_