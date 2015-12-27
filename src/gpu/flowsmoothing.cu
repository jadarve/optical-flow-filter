/**
 * \file flowsmoothing.cu
 * \brief Optical flow smoothing classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#include <iostream>
#include <exception>

#include "flowfilter/gpu/util.h"
#include "flowfilter/gpu/error.h"
#include "flowfilter/gpu/flowsmoothing.h"
#include "flowfilter/gpu/device/flowsmoothing_k.h"

namespace flowfilter {
namespace gpu {

FlowSmoother::FlowSmoother() : 
    Stage() {
    __configured = false;
    __inputFlowSet = false;
    __iterations = 0;
}


FlowSmoother::FlowSmoother(GPUImage inputFlow,
    const int iterations) :
    Stage() {

    __configured = false;
    __inputFlowSet = false;

    setInputFlow(inputFlow);
    setIterations(iterations);
    configure();
}


FlowSmoother::~FlowSmoother() {
    // nothing to do
}


void FlowSmoother::configure() {

    if(!__inputFlowSet) {
        std::cerr << "ERROR: FlowPropagator::configure(): input flow has not been set" << std::endl;
        throw std::exception();
    }

    int height = __inputFlow.height();
    int width = __inputFlow.width();

    // wrap __inputFlow in a texture
    __inputFlowTexture = GPUTexture(__inputFlow, cudaChannelFormatKindFloat);

    __smoothedFlow_X = GPUImage(height, width, 2, sizeof(float));
    __smoothedFlowTexture_X = GPUTexture(__smoothedFlow_X, cudaChannelFormatKindFloat);

    __smoothedFlow_Y = GPUImage(height, width, 2, sizeof(float));
    __smoothedFlowTexture_Y = GPUTexture(__smoothedFlow_Y, cudaChannelFormatKindFloat);


    // configure block and grid sizes
    __block = dim3(32, 32, 1);
    configureKernelGrid(height, width, __block, __grid);

    __configured = true;
}


void FlowSmoother::compute() {

    startTiming();

    if(!__configured) {
        std::cerr << "ERROR: FlowPropagator::compute() stage not configured." << std::endl;
        exit(-1);
    }

    // First iteration takes as input __inputFlow
    flowSmoothX_k<<<__grid, __block, 0, __stream>>>(
        __inputFlowTexture.getTextureObject(),
        __smoothedFlow_X.wrap<float2>());

    flowSmoothY_k<<<__grid, __block, 0, __stream>>>(
        __smoothedFlowTexture_X.getTextureObject(),
        __smoothedFlow_Y.wrap<float2>());


    // Rest of iterations
    for(int n = 0; n < __iterations - 1; n ++) {

        // take as input __smoothedFlowY
        flowSmoothX_k<<<__grid, __block, 0, __stream>>>(
            __smoothedFlowTexture_Y.getTextureObject(),
            __smoothedFlow_X.wrap<float2>());

        flowSmoothY_k<<<__grid, __block, 0, __stream>>>(
            __smoothedFlowTexture_X.getTextureObject(),
            __smoothedFlow_Y.wrap<float2>());
    }

    stopTiming();

}


int FlowSmoother::getIterations() const {

    return __iterations;
}


void FlowSmoother::setIterations(const int N) {

    if(N <= 0) {
        std::cerr << "ERROR: FlowSmoother::setIterations(): itersations should be greater than zero: "
            << N << std::endl;

        throw std::exception();
    }

    __iterations = N;
}


void FlowSmoother::setInputFlow(GPUImage inputFlow) {

    if(inputFlow.depth() != 2) {
        std::cerr << "ERROR: FlowSmoother::setInputFlow(): input flow should have depth 2: "
            << inputFlow.depth() << std::endl;
        throw std::exception();
    }

    if(inputFlow.itemSize() != 4) {
        std::cerr << "ERROR: FlowSmoother::setInputFlow(): input flow should have item size 4: "
            << inputFlow.itemSize() << std::endl;
        throw std::exception();
    }

    __inputFlow = inputFlow;
    __inputFlowSet = true;
}


GPUImage FlowSmoother::getSmoothedFlow() {

    return __smoothedFlow_Y;
}


}; // namespace gpu
}; // namespace flowfilter
