/**
 * \file update.cu
 * \brief Optical flow filter update classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <exception>

#include "flowfilter/gpu/util.h"
#include "flowfilter/gpu/error.h"
#include "flowfilter/gpu/update.h"
#include "flowfilter/gpu/device/update_k.h"

namespace flowfilter {
namespace gpu {


FlowUpdate::FlowUpdate() {

    __configured = false;
    __inputFlowSet = false;
    __inputImageSet = false;
    __inputImageGradientSet = false;
    __gamma = 1.0;
    __maxflow = 1.0;
}


FlowUpdate::FlowUpdate(flowfilter::gpu::GPUImage inputFlow,
           flowfilter::gpu::GPUImage inputImage,
           flowfilter::gpu::GPUImage inputImageGradient,
           const float gamma,
           const float maxflow) {

    __configured = false;
    __inputFlowSet = false;
    __inputImageSet = false;
    __inputImageGradientSet = false;
    
    setGamma(gamma);
    setMaxFlow(maxflow);
    setInputFlow(inputFlow);
    setInputImage(inputImage);
    setInputImageGradient(inputImageGradient);
    configure();
}


FlowUpdate::~FlowUpdate() {

    // nothing to do...
}


void FlowUpdate::configure() {

    if(!__inputFlowSet) {
        std::cerr << "ERROR: FlowUpdate::configure(): input flow not set" << std::endl;
        throw std::exception();
    }

    if(!__inputImageSet) {
        std::cerr << "ERROR: FlowUpdate::configure(): input image not set" << std::endl;
        throw std::exception();
    }

    if(!__inputImageGradientSet) {
        std::cerr << "ERROR: FlowUpdate::configure(): input image gradient not set" << std::endl;
        throw std::exception();
    }

    int height = __inputFlow.height();
    int width = __inputFlow.width();

    // verify that height and width of inputs are all the same
    if(height != __inputImage.height() || height != __inputImageGradient.height()
        || width != __inputImage.width() || width != __inputImageGradient.width()) {
        std::cerr << "ERROR: FlowUpdate::configure(): input buffers do not match height and width" << std::endl;
        throw std::exception();
    }

    __flowUpdated = GPUImage(height, width, 2, sizeof(float));
    __imageUpdated = GPUImage(height, width, 1, sizeof(float));

    // configure block and grid sizes
    __block = dim3(32, 32, 1);
    configureKernelGrid(height, width, __block, __grid);

    __configured = true;
}


void FlowUpdate::compute() {

    startTiming();

    flowUpdate_k<<<__grid, __block, 0, __stream>>>(
        __inputImage.wrap<float>(),
        __inputImageGradient.wrap<float2>(),
        __imageUpdated.wrap<float>(),
        __inputFlow.wrap<float2>(),
        __imageUpdated.wrap<float>(),
        __flowUpdated.wrap<float2>(),
        __gamma, __maxflow);

    stopTiming();
}


float FlowUpdate::getGamma() const {
    return __gamma;
}


void FlowUpdate::setGamma(const float gamma) {

    if(gamma <= 0) {
        std::cerr << "ERROR: FlowUpdate::setGamma(): gamma should be greater than zero: " << gamma << std::endl;
        throw std::exception();
    }

    __gamma = gamma;
}


float FlowUpdate::getMaxFlow() const {
    return __maxflow;
}


void FlowUpdate::setMaxFlow(const float maxflow) {

    __maxflow = maxflow;
}


void FlowUpdate::setInputFlow(flowfilter::gpu::GPUImage inputFlow) {

    if(inputFlow.depth() != 2) {
        std::cerr << "ERROR: FlowUpdate::setInputFlow(): input flow should have depth 2: "
            << inputFlow.depth() << std::endl;
        throw std::exception();
    }

    if(inputFlow.itemSize() != 4) {
        std::cerr << "ERROR: FlowUpdate::setInputFlow(): input flow should have item size 4: "
            << inputFlow.itemSize() << std::endl;
        throw std::exception();
    }

    __inputFlow = inputFlow;
    __inputFlowSet = true;
}


void FlowUpdate::setInputImage(flowfilter::gpu::GPUImage image) {

    if(image.depth() != 1) {
        std::cerr << "ERROR: FlowUpdate::setInputImage(): input image should have depth 1: "
            << image.depth() << std::endl;
        throw std::exception();
    }

    if(image.itemSize() != sizeof(float)) {
        std::cerr << "ERROR: FlowUpdate::setInputImage(): input image should have item size 4: "
            << image.itemSize() << std::endl;
        throw std::exception();
    }

    __inputImage = image;
    __inputImageSet = true;
}


void FlowUpdate::setInputImageGradient(flowfilter::gpu::GPUImage imageGradient) {

    if(imageGradient.depth() != 2) {
        std::cerr << "ERROR: FlowUpdate::setInputImageGradient(): input image gradient should have depth 2: "
            << imageGradient.depth() << std::endl;
        throw std::exception();
    }

    if(imageGradient.itemSize() != sizeof(float)) {
        std::cerr << "ERROR: FlowUpdate::setInputImageGradient(): input image gradient should have item size 4: "
            << imageGradient.itemSize() << std::endl;
        throw std::exception();
    }

    __inputImageGradient = imageGradient;
    __inputImageGradientSet = true;
}


flowfilter::gpu::GPUImage FlowUpdate::getUpdatedFlow() {

    return __flowUpdated;
}


flowfilter::gpu::GPUImage FlowUpdate::getUpdatedImage() {

    return __imageUpdated;
}

}; // namespace gpu
}; // namespace flowfilter