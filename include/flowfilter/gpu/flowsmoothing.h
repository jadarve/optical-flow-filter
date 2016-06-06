/**
 * \file flowsmoothing.h
 * \brief Optical flow smoothing classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_FLOWSMOOTHING_H_
#define FLOWFILTER_GPU_FLOWSMOOTHING_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/osconfig.h"
#include "flowfilter/gpu/pipeline.h"
#include "flowfilter/gpu/image.h"

namespace flowfilter {
namespace gpu {


class FLOWFILTER_API FlowSmoother : public Stage {

public:
    FlowSmoother();
    FlowSmoother(flowfilter::gpu::GPUImage inputFlow, const int iterations);
    ~FlowSmoother();

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

    int getIterations() const;
    void setIterations(const int N);

    //#########################
    // Stage inputs
    //#########################
    void setInputFlow(flowfilter::gpu::GPUImage inputFlow);

    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getSmoothedFlow();

private:

    int __iterations;

    /** tell if the stage has been configured */
    bool __configured;

    /** tells if an input flow has been set */
    bool __inputFlowSet;

    // inputs
    flowfilter::gpu::GPUImage __inputFlow;
    flowfilter::gpu::GPUTexture __inputFlowTexture;

    /** output of the smoothing in Y (row) direction */
    flowfilter::gpu::GPUImage __smoothedFlow_Y;
    flowfilter::gpu::GPUTexture __smoothedFlowTexture_Y;

    // intermediate buffers

    /** output of the smoothing in X (column) direction */
    flowfilter::gpu::GPUImage __smoothedFlow_X;
    flowfilter::gpu::GPUTexture __smoothedFlowTexture_X;

    // block and grid size for kernel calls
    dim3 __block;
    dim3 __grid;

};


}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_FLOWSMOOTHING_H_
