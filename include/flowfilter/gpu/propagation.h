/**
 * \file propagation.h
 * \brief Optical flow propagation classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_PROPAGATION_H_
#define FLOWFILTER_GPU_PROPAGATION_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/gpu/pipeline.h"
#include "flowfilter/gpu/image.h"

namespace flowfilter {
namespace gpu {


/**
 * \brief Optical flow propagator.
 */
class FlowPropagator : public Stage {

public:
    FlowPropagator();
    FlowPropagator(flowfilter::gpu::GPUImage inputFlow, const int iterations=1);
    ~FlowPropagator();

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

    void setIterations(const int N);
    int getIterations() const;
    float getDt() const;

    //#########################
    // Stage inputs
    //#########################
    void setInputFlow(flowfilter::gpu::GPUImage inputFlow);

    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getPropagatedFlow();


private:

    int __iterations;
    float __dt;

    /** tell if the stage has been configured */
    bool __configured;

    /** tells if an input flow has been set */
    bool __inputFlowSet;

    // inputs
    flowfilter::gpu::GPUImage __inputFlow;
    flowfilter::gpu::GPUTexture __inputFlowTexture;

    // outputs

    /** output of the propagation in Y (row) direction */
    flowfilter::gpu::GPUImage __propagatedFlow_Y;
    flowfilter::gpu::GPUTexture __propagatedFlowTexture_Y;


    // intermediate buffers

    /** output of the propagation in X (column) direction */
    flowfilter::gpu::GPUImage __propagatedFlow_X;
    flowfilter::gpu::GPUTexture __propagatedFlowTexture_X;


    // block and grid size for kernel calls
    dim3 __block;
    dim3 __grid;
};


// class FlowPropagatorPayload : public Stage {

// public:

//     /**
//      * \brief configures the stage.
//      *
//      * After configuration, calls to compute()
//      * are valid.
//      * Input buffers should not change after
//      * this method has been called.
//      */
//     void configure();

//     /**
//      * \brief performs computation of brightness parameters
//      */
//     void compute();

//     //#########################
//     // Stage inputs
//     //#########################
//     void setIterations(const int N);
//     void setInputFlow(flowfilter::gpu::GPUImage& img);
//     void setPayloadScalar(flowfilter::gpu::GPUImage& img);
//     void setPayloadVector2(flowfilter::gpu::GPUImage& img);

//     //#########################
//     // Stage outputs
//     //#########################
//     flowfilter::gpu::GPUImage getPropagatedFlow();
//     flowfilter::gpu::GPUImage getPropagatedPayloadScalar();
//     flowfilter::gpu::GPUImage getPropagatedPayloadVector2();
// };


}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_PROPAGATION_H_