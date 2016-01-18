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


FlowUpdate::FlowUpdate() :
    Stage() {

    __configured = false;
    __inputFlowSet = false;
    __inputImageSet = false;
    __inputImageGradientSet = false;
    __gamma = 1.0;
    __maxflow = 1.0;
}


FlowUpdate::FlowUpdate(GPUImage inputFlow,
           GPUImage inputImage,
           GPUImage inputImageGradient,
           const float gamma,
           const float maxflow) : 
    Stage() {

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

    if(!__configured) {
        std::cerr << "ERROR: FlowUpdate::compute() stage not configured." << std::endl;
        exit(-1);
    }

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


void FlowUpdate::setInputFlow(GPUImage inputFlow) {

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


void FlowUpdate::setInputImage(GPUImage image) {

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


void FlowUpdate::setInputImageGradient(GPUImage imageGradient) {

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


GPUImage FlowUpdate::getUpdatedFlow() {

    return __flowUpdated;
}


GPUImage FlowUpdate::getUpdatedImage() {

    return __imageUpdated;
}




//###############################################
// DeltaFlowUpdate
//###############################################

DeltaFlowUpdate::DeltaFlowUpdate() :
    Stage() {

    __gamma = 1.0;
    __maxflow = 1.0;
    __configured = false;
    __inputDeltaFlowSet = false;
    __inputImageOldSet = false;
    __inputFlowSet = false;
    __inputImageSet = false;
    __inputImageGradientSet = false;
}


DeltaFlowUpdate::DeltaFlowUpdate(GPUImage inputFlow,
    GPUImage inputDeltaFlow,
    GPUImage inputImageOld,
    GPUImage inputImage,
    GPUImage inputImageGradient,
    const float gamma,
    const float maxflow) :
    Stage() {

    __configured = false;
    __inputDeltaFlowSet = false;
    __inputImageOldSet = false;
    __inputFlowSet = false;
    __inputImageSet = false;
    __inputImageGradientSet = false;

    setGamma(gamma);
    setMaxFlow(maxflow);
    setInputDeltaFlow(inputDeltaFlow);
    setInputFlow(inputFlow);
    setInputImageOld(inputImageOld);
    setInputImage(inputImage);
    setInputImageGradient(inputImageGradient);
    configure();
}

DeltaFlowUpdate::~DeltaFlowUpdate() {

    // nothing to do
}


void DeltaFlowUpdate::configure() {

    if(!__inputFlowSet) {
        std::cerr << "ERROR: DeltaFlowUpdate::configure(): input flow not set" << std::endl;
        throw std::exception();
    }

    if(!__inputDeltaFlowSet) {
        std::cerr << "ERROR: DeltaFlowUpdate::configure(): input delta flow not set" << std::endl;
        throw std::exception();
    }

    if(!__inputImageOldSet) {
        std::cerr << "ERROR: DeltaFlowUpdate::configure(): input image prior not set" << std::endl;
        throw std::exception();
    }

    if(!__inputImageSet) {
        std::cerr << "ERROR: DeltaFlowUpdate::configure(): input image not set" << std::endl;
        throw std::exception();
    }

    if(!__inputImageGradientSet) {
        std::cerr << "ERROR: DeltaFlowUpdate::configure(): input image gradient not set" << std::endl;
        throw std::exception();
    }

    int height = __inputDeltaFlow.height();
    int width = __inputDeltaFlow.width();

    // verify that height and width of inputs are all the same
    if( height != __inputImage.height() || width != __inputImage.width() ||
        height != __inputImageGradient.height() || width != __inputImageGradient.width() ||
        height != __inputImageOld.height() || width != __inputImageOld.width()) {

        std::cerr << "ERROR: DeltaFlowUpdate::configure(): input buffers do not match height and width" << std::endl;
        throw std::exception();
    }
    
    // configure texture for reading inputFlow
    // using normalized texture coordinates and linear interpolation
    __inputFlowTexture = GPUTexture(__inputFlow,
        cudaChannelFormatKindFloat,
        cudaAddressModeClamp,
        cudaFilterModePoint,
        cudaReadModeElementType,
        true);

    // outputs
    __flowUpdated = GPUImage(height, width, 2, sizeof(float));
    __deltaFlowUpdated = GPUImage(height, width, 2, sizeof(float));
    __imageUpdated = GPUImage(height, width, 1, sizeof(float));

    // configure block and grid sizes
    __block = dim3(32, 32, 1);
    configureKernelGrid(height, width, __block, __grid);

    __configured = true;

}

void DeltaFlowUpdate::compute() {

    startTiming();

    if(!__configured) {
        std::cerr << "ERROR: DeltaFlowUpdate::compute() stage not configured." << std::endl;
        exit(-1);
    }

    deltaFlowUpdate_k<<<__grid, __block, 0, __stream>>> (
        __inputImage.wrap<float>(),
        __inputImageGradient.wrap<float2>(),
        __inputImageOld.wrap<float>(),
        __inputDeltaFlow.wrap<float2>(),
        __inputFlowTexture.getTextureObject(),
        __imageUpdated.wrap<float>(),
        __deltaFlowUpdated.wrap<float2>(),
        __flowUpdated.wrap<float2>(),
        __gamma, __maxflow);

    stopTiming();
}

float DeltaFlowUpdate::getGamma() const {
    return __gamma;
}

void DeltaFlowUpdate::setGamma(const float gamma) {

    if(gamma <= 0) {
        std::cerr << "ERROR: DeltaFlowUpdate::setGamma(): gamma should be greater than zero: " << gamma << std::endl;
        throw std::exception();
    }

    __gamma = gamma;
}


float DeltaFlowUpdate::getMaxFlow() const {
    return __maxflow;
}


void DeltaFlowUpdate::setMaxFlow(const float maxflow) {

    __maxflow = maxflow;
}


void DeltaFlowUpdate::setInputFlow(GPUImage inputFlow) {

    if(inputFlow.depth() != 2) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputFlow(): input flow should have depth 2: "
            << inputFlow.depth() << std::endl;
        throw std::exception();
    }

    if(inputFlow.itemSize() != 4) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputFlow(): input flow should have item size 4: "
            << inputFlow.itemSize() << std::endl;
        throw std::exception();
    }

    __inputFlow = inputFlow;
    __inputFlowSet = true;
}


void DeltaFlowUpdate::setInputDeltaFlow(GPUImage inputDeltaFlow) {

    if(inputDeltaFlow.depth() != 2) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputDeltaFlow(): input delta flow should have depth 2: "
            << inputDeltaFlow.depth() << std::endl;
        throw std::exception();
    }

    if(inputDeltaFlow.itemSize() != 4) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputFlow(): input flow should have item size 4: "
            << inputDeltaFlow.itemSize() << std::endl;
        throw std::exception();
    }

    __inputDeltaFlow = inputDeltaFlow;
    __inputDeltaFlowSet = true;
}

void DeltaFlowUpdate::setInputImageOld(GPUImage image) {

    if(image.depth() != 1) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputImageOld(): input image should have depth 1: "
            << image.depth() << std::endl;
        throw std::exception();
    }

    if(image.itemSize() != sizeof(float)) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputImageOld(): input image should have item size 4: "
            << image.itemSize() << std::endl;
        throw std::exception();
    }

    __inputImageOld = image;
    __inputImageOldSet = true;
}


void DeltaFlowUpdate::setInputImage(GPUImage image) {

    if(image.depth() != 1) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputImage(): input image should have depth 1: "
            << image.depth() << std::endl;
        throw std::exception();
    }

    if(image.itemSize() != sizeof(float)) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputImage(): input image should have item size 4: "
            << image.itemSize() << std::endl;
        throw std::exception();
    }

    __inputImage = image;
    __inputImageSet = true;
}


void DeltaFlowUpdate::setInputImageGradient(GPUImage imageGradient) {

    if(imageGradient.depth() != 2) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputImageGradient(): input image gradient should have depth 2: "
            << imageGradient.depth() << std::endl;
        throw std::exception();
    }

    if(imageGradient.itemSize() != sizeof(float)) {
        std::cerr << "ERROR: DeltaFlowUpdate::setInputImageGradient(): input image gradient should have item size 4: "
            << imageGradient.itemSize() << std::endl;
        throw std::exception();
    }

    __inputImageGradient = imageGradient;
    __inputImageGradientSet = true;
}


GPUImage DeltaFlowUpdate::getUpdatedFlow() {

    return __flowUpdated;
}

GPUImage DeltaFlowUpdate::getUpdatedDeltaFlow() {

    return __deltaFlowUpdated;
}


GPUImage DeltaFlowUpdate::getUpdatedImage() {

    return __imageUpdated;
}

}; // namespace gpu
}; // namespace flowfilter