/**
 * \file propagation.cu
 * \brief Optical flow propagation classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <exception>

#include "flowfilter/gpu/util.h"
#include "flowfilter/gpu/error.h"
#include "flowfilter/gpu/propagation.h"
#include "flowfilter/gpu/device/propagation_k.h"

namespace flowfilter {
namespace gpu {

FlowPropagator::FlowPropagator() :
    Stage() {

    __configured = false;
    __inputFlowSet = false;
    __iterations = 0;
    __dt = 0.0f;
}

FlowPropagator::FlowPropagator(GPUImage& inputFlow,
    const int iterations) : 
    Stage() {

    __configured = false;
    __inputFlowSet = false;

    setInputFlow(inputFlow);
    setIterations(iterations);
    configure();
}

FlowPropagator::~FlowPropagator() {
    // nothing to do...
}

void FlowPropagator::configure() {

    if(!__inputFlowSet) {
        std::cerr << "ERROR: FlowPropagator::configure(): input flow has not been set" << std::endl;
        throw std::exception();
    }

    int height = __inputFlow.height();
    int width = __inputFlow.width();

    // wrap __inputFlow in a texture
    __inputFlowTexture = GPUTexture(__inputFlow, cudaChannelFormatKindFloat);

    __propagatedFlow_X = GPUImage(height, width, 2, sizeof(float));
    __propagatedFlowTexture_X = GPUTexture(__propagatedFlow_X, cudaChannelFormatKindFloat);

    __propagatedFlow_Y = GPUImage(height, width, 2, sizeof(float));
    __propagatedFlowTexture_Y = GPUTexture(__propagatedFlow_Y, cudaChannelFormatKindFloat);


    // configure block and grid sizes
    __block = dim3(32, 32, 1);
    configureKernelGrid(height, width, __block, __grid);

    __configured = true;
}

void FlowPropagator::compute() {

    startTiming();

    if(!__configured) {
        std::cerr << "ERROR: FlowPropagator::compute() stage not configured." << std::endl;
        exit(-1);
    }

    // First iteration takes as input __inputFlow
    flowPropagateX_k<<<__grid, __block, 0, __stream>>>(
        __inputFlowTexture.getTextureObject(),
        __propagatedFlow_X.wrap<float2>(), __dt, 1);

    flowPropagateY_k<<<__grid, __block, 0, __stream>>>(
        __propagatedFlowTexture_X.getTextureObject(),
        __propagatedFlow_Y.wrap<float2>(), __dt, 1);


    // Rest of iterations
    for(int n = 0; n < __iterations - 1; n ++) {

        // take as input __propagatedFlowY
        flowPropagateX_k<<<__grid, __block, 0, __stream>>>(
            __propagatedFlowTexture_Y.getTextureObject(),
            __propagatedFlow_X.wrap<float2>(), __dt, 1);

        flowPropagateY_k<<<__grid, __block, 0, __stream>>>(
            __propagatedFlowTexture_X.getTextureObject(),
            __propagatedFlow_Y.wrap<float2>(), __dt, 1);
    }

    stopTiming();
}

void FlowPropagator::setIterations(const int N) {

    if(N <= 0) {
        std::cerr << "ERROR: FlowPropagator::setIterations(): iterations less than zero: "
            << N << std::endl;

        throw std::exception();
    }

    __iterations = N;
    __dt = 1.0f / float(__iterations);
}

int FlowPropagator::getIterations() const {
    return __iterations;
}

float FlowPropagator::getDt() const {
    return __dt;
}

void FlowPropagator::setInputFlow(flowfilter::gpu::GPUImage& inputFlow) {

    if(inputFlow.depth() != 2) {
        std::cerr << "ERROR: FlowPropagator::setInputFlow(): input flow should have depth 2: "
            << inputFlow.depth() << std::endl;
        throw std::exception();
    }

    if(inputFlow.itemSize() != 4) {
        std::cerr << "ERROR: FlowPropagator::setInputFlow(): input flow should have item size 4: "
            << inputFlow.itemSize() << std::endl;
        throw std::exception();
    }

    __inputFlow = inputFlow;
    __inputFlowSet = true;

}

GPUImage FlowPropagator::getPropagatedFlow() {

    return __propagatedFlow_Y;
}

}; // namespace gpu
}; // namespace flowfilter
