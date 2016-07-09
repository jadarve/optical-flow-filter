/**
 * \file display.h
 * \brief Contain classes to color encode Optical flow fields.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_DISPLAY_H_
#define FLOWFILTER_DISPLAY_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/osconfig.h"

#include "flowfilter/gpu/pipeline.h"
#include "flowfilter/gpu/image.h"


namespace flowfilter {
namespace gpu {

class FLOWFILTER_API FlowToColor : public Stage {

public:
    FlowToColor();
    FlowToColor(flowfilter::gpu::GPUImage inputFlow, const float maxflow);
    ~FlowToColor();

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


    //#########################
    // Host load-download
    //#########################

    /**
     * \brief download the RGBA color encoding of optical flow
     */
    void downloadColorFlow(flowfilter::image_t& colorFlow);


    //#########################
    // Stage inputs
    //#########################
    void setInputFlow(flowfilter::gpu::GPUImage inputFlow);


    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getColorFlow();


    //#########################
    // Parameters
    //#########################
    float getMaxFlow() const;
    void setMaxFlow(const float maxflow);

private:
    bool __configured;
    bool __inputFlowSet;

    float __maxflow;

    flowfilter::gpu::GPUImage __colorWheel;
    flowfilter::gpu::GPUTexture __colorWheelTexture;

    // inputs
    flowfilter::gpu::GPUImage __inputFlow;

    // outputs
    flowfilter::gpu::GPUImage __colorFlow;

    dim3 __block;
    dim3 __grid;
};

}; // namepsace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_DISPLAY_H_