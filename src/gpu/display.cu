/**
 * \file display.cu
 * \brief Contain classes to color encode Optical flow fields.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <exception>
#include <iostream>

#include "flowfilter/image.h"
#include "flowfilter/colorwheel.h"
#include "flowfilter/gpu/util.h"
#include "flowfilter/gpu/display.h"
#include "flowfilter/gpu/device/display_k.h"


namespace flowfilter {
namespace gpu {


FlowToColor::FlowToColor() :
    Stage() {

    __configured = false;
    __inputFlowSet = false;
    __maxflow = 1.0f;
}

FlowToColor::FlowToColor(flowfilter::gpu::GPUImage inputFlow, 
    const float maxflow) :
    Stage() {

    __configured = false;
    __inputFlowSet = false;

    setInputFlow(inputFlow);
    setMaxFlow(maxflow);
    configure();
}


FlowToColor::~FlowToColor() {
    // nothing to do
}


void FlowToColor::configure() {

    if(!__inputFlowSet) {
        std::cerr << "ERROR: FlowToColor::configure(): input flow not set" << std::endl;
        throw std::exception();
    }

    // creates an RGBA images from the RGB color wheel
    image_t wheelRGBA = getColorWheelRGBA();

    __colorWheel = GPUImage(wheelRGBA.height,
        wheelRGBA.width, wheelRGBA.depth, sizeof(unsigned char));

    // upload RGBA color to device
    __colorWheel.upload(wheelRGBA);

    // configure texture to read uchar4 with normalized coordinates
    __colorWheelTexture = GPUTexture(__colorWheel,
        cudaChannelFormatKindUnsigned, cudaReadModeElementType, true);

    // output coloured optical flow
    __colorFlow = GPUImage(__inputFlow.height(), __inputFlow.width(), 4, sizeof(unsigned char));


    // configure block and grid sizes
    __block = dim3(32, 32, 1);
    configureKernelGrid(__inputFlow.height(), __inputFlow.width(),
        __block, __grid);

    __configured = true;
}


void FlowToColor::compute() {
    
    startTiming();

    if(!__configured) {
        std::cerr << "ERROR: FlowToColor::compute(): Stage not configured" << std::endl;
        throw std::exception();
    }

    flowToColor_k<<<__grid, __block, 0, __stream>>>(
        __inputFlow.wrap<float2>(), __colorWheelTexture.getTextureObject(),
        __maxflow, __colorFlow.wrap<uchar4>()
        );


    stopTiming();
}


void FlowToColor::setInputFlow(GPUImage inputFlow) {

    if(inputFlow.depth() != 2) {
        std::cerr << "ERROR: FlowToColor::setInputFlow(): input flow should have depth 2: "
            << inputFlow.depth() << std::endl;
        throw std::exception();
    }

    if(inputFlow.itemSize() != 4) {
        std::cerr << "ERROR: FlowToColor::setInputFlow(): input flow should have item size 4: "
            << inputFlow.itemSize() << std::endl;
        throw std::exception();
    }

    __inputFlow = inputFlow;
    __inputFlowSet = true;
}


GPUImage FlowToColor::getColorFlow() {
    return __colorFlow;
}


float FlowToColor::getMaxFlow() const {
    return __maxflow;
}


void FlowToColor::setMaxFlow(const float maxflow) {

    if(maxflow <= 0.0f) {
        std::cerr << "ERROR: FlowToColor::setMaxFlow(): maxflow should be greater than 0.0: " << maxflow << std::endl;
        throw std::exception();
    }

    __maxflow = maxflow;
}


void FlowToColor::downloadColorFlow(flowfilter::image_t& colorFlow) {
    __colorFlow.download(colorFlow);
}


}; // namespace gpu
}; // namespace flowfilter