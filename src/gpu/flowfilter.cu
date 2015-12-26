/**
 * \file flowfilter.cu
 * \brief Optical flow filter classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#include <iostream>
#include <exception>

#include "flowfilter/gpu/util.h"
#include "flowfilter/gpu/error.h"
#include "flowfilter/gpu/flowfilter.h"

namespace flowfilter {
namespace gpu {


FlowFilter::FlowFilter() :
    Stage() {

}

FlowFilter::FlowFilter(const int height, const int witdh) :
    Stage() {

}

FlowFilter::~FlowFilter() {
    // nothing to do
}


void FlowFilter::configure() {

    // connect the blocks
    __inputImage = GPUImage(__height, __width, 1, sizeof(unsigned char));
    __imageModel = ImageModel(__inputImage);

    // // dummy flow field use to instanciate the update block
    // // This is necessary to break the circular dependency
    // // between propagation and update blocks.
    // GPUImage dummyFlow(__height, __width, 2, sizeof(float));

    // // FIXME: problems with arguments by reference!

    // __update = FlowUpdate(dummyFlow,
    //     __imageModel.getImageConstant().
    //     __imageModel.getImageGradient(),
    //     __gamma, __maxflow);

    // __smoother = FlowSmoother(__update.getUpdatedFlow(), __smoothIterations);

    // __propagator = FlowPropagator(__smoother.getFlow(), __propagationIterations);

    // // set the input flow of the update block to the output
    // // of the propagator. This replaces dummyFlow previously
    // // assigned to the update
    // __update.setInputFlow(__propagator.getPropagatedFlow());

    __configured = true;
    __firstLoad = true;
}

void FlowFilter::compute() {

    startTiming();

    // compute image model
    __imageModel.compute();

    // propagate old flow
    __propagator.compute();

    // update
    __update.compute();

    // smooth updated flow
    __smoother.compute();

    stopTiming();
}

void FlowFilter::loadImage(flowfilter::image_t& image) {

}

void FlowFilter::downloadFlow(flowfilter::image_t& flow) {

}

GPUImage FlowFilter::getFlow() {
    return __update.getUpdatedFlow();
}


}; // namespace gpu
}; // namespace flowfilter