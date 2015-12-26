/**
 * \file flowfilter.h
 * \brief Optical flow filter classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_FLOWFILTER_H_
#define FLOWFILTER_GPU_FLOWFILTER_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/image.h"

#include "flowfilter/gpu/image.h"
#include "flowfilter/gpu/pipeline.h"
#include "flowfilter/gpu/imagemodel.h"
#include "flowfilter/gpu/update.h"
#include "flowfilter/gpu/propagation.h"
#include "flowfilter/gpu/flowsmoothing.h"


namespace flowfilter {
namespace gpu {

class FlowFilter : public Stage {

public:
    FlowFilter();
    FlowFilter(const int height, const int witdh);
    ~FlowFilter();

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

    /**
     * \brief load image stored in CPU memory space
     */
    void loadImage(flowfilter::image_t& image);

    void downloadFlow(flowfilter::image_t& flow);

    flowfilter::gpu::GPUImage getFlow();


private:
    int __height;
    int __width;

    bool __configured;
    bool __firstLoad;

    float __gamma;
    float __maxflow;
    
    int __propagationIterations;
    int __smoothIterations;

    flowfilter::gpu::GPUImage __inputImage;

    flowfilter::gpu::ImageModel __imageModel;
    flowfilter::gpu::FlowUpdate __update;
    flowfilter::gpu::FlowSmoother __smoother;
    flowfilter::gpu::FlowPropagator __propagator;

};

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_FLOWFILTER_H_