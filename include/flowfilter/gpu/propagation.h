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

#include "flowfilter/osconfig.h"
#include "flowfilter/gpu/pipeline.h"
#include "flowfilter/gpu/image.h"

namespace flowfilter {
namespace gpu {


/**
 * \brief Optical flow propagator.
 */
class FLOWFILTER_API FlowPropagator : public Stage {

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

    void setBorder(const int border);
    int getBorder() const;

    void setInvertInputFlow(const bool invert);
    bool getInvertInputFlow() const;

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
    int __border;

    /** tell if the stage has been configured */
    bool __configured;

    /** tells if an input flow has been set */
    bool __inputFlowSet;

    bool __invertInputFlow;

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


/**
 * \brief Optical flow propagator with scalar and vector payloads.
 */
class FLOWFILTER_API FlowPropagatorPayload : public Stage {

public:
    FlowPropagatorPayload();
    FlowPropagatorPayload(flowfilter::gpu::GPUImage inputFlow,
        flowfilter::gpu::GPUImage scalarPayload,
        flowfilter::gpu::GPUImage vectorPayload,
        const int iterations=1);

    ~FlowPropagatorPayload();

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

    void setBorder(const int border);
    int getBorder() const;

    //#########################
    // Stage inputs
    //#########################
    void setInputFlow(flowfilter::gpu::GPUImage img);
    void setScalarPayload(flowfilter::gpu::GPUImage img);
    void setVectorPayload(flowfilter::gpu::GPUImage img);

    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getPropagatedFlow();
    flowfilter::gpu::GPUImage getPropagatedScalar();
    flowfilter::gpu::GPUImage getPropagatedVector();


private:

    int __iterations;
    float __dt;
    int __border;

    /** tell if the stage has been configured */
    bool __configured;

    /** tells if an input flow has been set */
    bool __inputFlowSet;
    bool __scalarPayloadSet;
    bool __vectorPayloadSet;

    // inputs
    flowfilter::gpu::GPUImage __inputFlow;
    flowfilter::gpu::GPUTexture __inputFlowTexture;

    flowfilter::gpu::GPUImage __inputScalar;
    flowfilter::gpu::GPUTexture __inputScalarTexture;

    flowfilter::gpu::GPUImage __inputVector;
    flowfilter::gpu::GPUTexture __inputVectorTexture;

    // outputs

    /** output of the propagation in Y (row) direction */
    flowfilter::gpu::GPUImage __propagatedFlow_Y;
    flowfilter::gpu::GPUTexture __propagatedFlowTexture_Y;

    flowfilter::gpu::GPUImage __propagatedScalar_Y;
    flowfilter::gpu::GPUTexture __propagatedScalarTexture_Y;

    flowfilter::gpu::GPUImage __propagatedVector_Y;
    flowfilter::gpu::GPUTexture __propagatedVectorTexture_Y;


    // intermediate buffers
    
    /** output of the propagation in X (column) direction */
    flowfilter::gpu::GPUImage __propagatedFlow_X;
    flowfilter::gpu::GPUTexture __propagatedFlowTexture_X;

    flowfilter::gpu::GPUImage __propagatedScalar_X;
    flowfilter::gpu::GPUTexture __propagatedScalarTexture_X;

    flowfilter::gpu::GPUImage __propagatedVector_X;
    flowfilter::gpu::GPUTexture __propagatedVectorTexture_X;


    // block and grid size for kernel calls
    dim3 __block;
    dim3 __grid;
};



class FLOWFILTER_API LaxWendroffPropagator : public Stage {

public:
    LaxWendroffPropagator();
    LaxWendroffPropagator(flowfilter::gpu::GPUImage inputFlow,
        flowfilter::gpu::GPUImage inputImage);
    ~LaxWendroffPropagator();

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
    // Parameters
    //#########################
    void setIterations(const int N);
    int getIterations() const;
    float getDt() const;

    //#########################
    // Stage inputs
    //#########################
    void setInputFlow(flowfilter::gpu::GPUImage inputFlow);
    void setInputImage(flowfilter::gpu::GPUImage img);

    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getFlow();
    flowfilter::gpu::GPUImage getPropagatedImage();


private:
    int __iterations;
    float __dt;

    /** tell if the stage has been configured */
    bool __configured;

    /** tells if an input flow has been set */
    bool __inputFlowSet;
    bool __inputImageSet;

    // inputs
    flowfilter::gpu::GPUImage __inputFlow;
    flowfilter::gpu::GPUTexture __inputFlowTexture;

    flowfilter::gpu::GPUImage __inputImage;
    flowfilter::gpu::GPUTexture __inputImageTexture;

    // outputs
    flowfilter::gpu::GPUImage __propagatedImage_X;
    flowfilter::gpu::GPUTexture __propagatedImageTexture_X;    

    // intermediate
    flowfilter::gpu::GPUImage __propagatedImage_Y;
    flowfilter::gpu::GPUTexture __propagatedImageTexture_Y;

    // block and grid size for kernel calls
    dim3 __block;
    dim3 __grid;
};

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_PROPAGATION_H_
