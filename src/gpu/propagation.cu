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
#include "flowfilter/gpu/device/misc_k.h"

namespace flowfilter {
namespace gpu {

FlowPropagator::FlowPropagator() :
    Stage() {

    __configured = false;
    __inputFlowSet = false;
    __invertInputFlow = false;
    __iterations = 0;
    __border = 3;
    __dt = 0.0f;
}


FlowPropagator::FlowPropagator(GPUImage inputFlow,
    const int iterations) : 
    Stage() {

    __configured = false;
    __inputFlowSet = false;
    __invertInputFlow = false;
    __border = 3;

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

    //#######################
    // First Iteration
    //#######################
    if(__invertInputFlow) {

        // invert __inputFlow and write it to __propagatedFlow_Y
        scalarProductF2_k<<<__grid, __block, 0, __stream>>>(
            __inputFlow.wrap<float2>(), -1.0f,
            __propagatedFlow_Y.wrap<float2>());

        // propagate in X using inverted flow written in __propagatedFlow_Y
        flowPropagateX_k<<<__grid, __block, 0, __stream>>>(
            __propagatedFlowTexture_Y.getTextureObject(),
            __propagatedFlow_X.wrap<float2>(), __dt, __border);

    } else {

        // Iterate in X using __inputFlow directly
        flowPropagateX_k<<<__grid, __block, 0, __stream>>>(
            __inputFlowTexture.getTextureObject(),
            __propagatedFlow_X.wrap<float2>(), __dt, __border);
    }

    // first iteration in Y
    flowPropagateY_k<<<__grid, __block, 0, __stream>>>(
        __propagatedFlowTexture_X.getTextureObject(),
        __propagatedFlow_Y.wrap<float2>(), __dt, __border);


    //#######################
    // Rest of iterations
    //#######################
    for(int n = 0; n < __iterations - 1; n ++) {

        // take as input __propagatedFlowY
        flowPropagateX_k<<<__grid, __block, 0, __stream>>>(
            __propagatedFlowTexture_Y.getTextureObject(),
            __propagatedFlow_X.wrap<float2>(), __dt, __border);

        flowPropagateY_k<<<__grid, __block, 0, __stream>>>(
            __propagatedFlowTexture_X.getTextureObject(),
            __propagatedFlow_Y.wrap<float2>(), __dt, __border);
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

void FlowPropagator::setBorder(const int border) {

    if(border < 0) {
        std::cerr << "ERROR: FlowPropagator::setBorder(): border should be greater of equal zero: "
            << border << std::endl;

        throw std::exception();
    }

    __border = border;
}


int FlowPropagator::getBorder() const {
    return __border;
}


void FlowPropagator::setInputFlow(flowfilter::gpu::GPUImage inputFlow) {

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


void FlowPropagator::setInvertInputFlow(const bool invert) {
    __invertInputFlow = invert;
}


bool FlowPropagator::getInvertInputFlow() const {
    return __invertInputFlow;
}


//###############################################
// FlowPropagatorPayload
//###############################################
FlowPropagatorPayload::FlowPropagatorPayload() :
    Stage() {

    __iterations = 0;
    __dt = 0.0f;
    __border = 3;
    __configured = false;

    __inputFlowSet = false;
    __scalarPayloadSet = false;
    __vectorPayloadSet = false;

}


FlowPropagatorPayload::FlowPropagatorPayload(GPUImage inputFlow,
    GPUImage scalarPayload,
    GPUImage vectorPayload,
    const int iterations) :
    Stage() {

    __iterations = 0;
    __dt = 0.0f;
    __border = 3;
    __configured = false;

    __inputFlowSet = false;
    __scalarPayloadSet = false;
    __vectorPayloadSet = false;

    setInputFlow(inputFlow);
    setScalarPayload(scalarPayload);
    setVectorPayload(vectorPayload);
    setIterations(iterations);
    configure();
}


FlowPropagatorPayload::~FlowPropagatorPayload() {

    // nothing to do
}



void FlowPropagatorPayload::configure() {

    if(!__inputFlowSet) {
        std::cerr << "ERROR: FlowPropagatorPayload::configure(): input flow has not been set" << std::endl;
        throw std::exception();
    }

    if(!__scalarPayloadSet) {
        std::cerr << "ERROR: FlowPropagatorPayload::configure(): input scalar payload has not been set" << std::endl;
        throw std::exception();
    }

    if(!__vectorPayloadSet) {
        std::cerr << "ERROR: FlowPropagatorPayload::configure(): input vector payload has not been set" << std::endl;
        throw std::exception();
    }

    int height = __inputFlow.height();
    int width = __inputFlow.width();

    //##################
    // flow
    //##################
    __inputFlowTexture = GPUTexture(__inputFlow, cudaChannelFormatKindFloat);

    __propagatedFlow_X = GPUImage(height, width, 2, sizeof(float));
    __propagatedFlowTexture_X = GPUTexture(__propagatedFlow_X, cudaChannelFormatKindFloat);

    __propagatedFlow_Y = GPUImage(height, width, 2, sizeof(float));
    __propagatedFlowTexture_Y = GPUTexture(__propagatedFlow_Y, cudaChannelFormatKindFloat);

    //##################
    // scalar payload
    //##################
    __inputScalarTexture = GPUTexture(__inputScalar, cudaChannelFormatKindFloat);

    __propagatedScalar_X = GPUImage(height, width, 1, sizeof(float));
    __propagatedScalarTexture_X = GPUTexture(__propagatedScalar_X, cudaChannelFormatKindFloat);

    __propagatedScalar_Y = GPUImage(height, width, 1, sizeof(float));
    __propagatedScalarTexture_Y = GPUTexture(__propagatedScalar_Y, cudaChannelFormatKindFloat);

    //##################
    // vector payload
    //##################
    __inputVectorTexture = GPUTexture(__inputVector, cudaChannelFormatKindFloat);

    __propagatedVector_X = GPUImage(height, width, 2, sizeof(float));
    __propagatedVectorTexture_X = GPUTexture(__propagatedVector_X, cudaChannelFormatKindFloat);

    __propagatedVector_Y = GPUImage(height, width, 2, sizeof(float));
    __propagatedVectorTexture_Y = GPUTexture(__propagatedVector_Y, cudaChannelFormatKindFloat);


    // configure block and grid sizes
    __block = dim3(32, 32, 1);
    configureKernelGrid(height, width, __block, __grid);

    __configured = true;
}


void FlowPropagatorPayload::compute() {

    startTiming();

    if(!__configured) {
        std::cerr << "ERROR: FlowPropagator::compute() stage not configured." << std::endl;
        exit(-1);
    }

    // First iteration takes as input __inputFlow
    flowPropagatePayloadX_k<<<__grid, __block, 0, __stream>>>(
        __inputFlowTexture.getTextureObject(),
        __propagatedFlow_X.wrap<float2>(),
        __inputScalarTexture.getTextureObject(),
        __propagatedScalar_X.wrap<float>(),
        __inputVectorTexture.getTextureObject(),
        __propagatedVector_X.wrap<float2>(), __dt, __border);

    flowPropagatePayloadY_k<<<__grid, __block, 0, __stream>>>(
        __propagatedFlowTexture_X.getTextureObject(),
        __propagatedFlow_Y.wrap<float2>(),
        __propagatedScalarTexture_X.getTextureObject(),
        __propagatedScalar_Y.wrap<float>(),
        __propagatedVectorTexture_X.getTextureObject(),
        __propagatedVector_Y.wrap<float2>(), __dt, __border);


    // Rest of iterations
    for(int n = 0; n < __iterations - 1; n ++) {

        // take as input __propagatedFlowY
        flowPropagatePayloadX_k<<<__grid, __block, 0, __stream>>>(
            __propagatedFlowTexture_Y.getTextureObject(),
            __propagatedFlow_X.wrap<float2>(),
            __propagatedScalarTexture_Y.getTextureObject(),
            __propagatedScalar_X.wrap<float>(),
            __propagatedVectorTexture_Y.getTextureObject(),
            __propagatedVector_X.wrap<float2>(), __dt, __border);

        flowPropagatePayloadY_k<<<__grid, __block, 0, __stream>>>(
            __propagatedFlowTexture_X.getTextureObject(),
            __propagatedFlow_Y.wrap<float2>(),
            __propagatedScalarTexture_X.getTextureObject(),
            __propagatedScalar_Y.wrap<float>(),
            __propagatedVectorTexture_X.getTextureObject(),
            __propagatedVector_Y.wrap<float2>(), __dt, __border);
    }

    stopTiming();
}


void FlowPropagatorPayload::setIterations(const int N) {

    if(N <= 0) {
        std::cerr << "ERROR: FlowPropagatorPayload::setIterations(): iterations less than zero: "
            << N << std::endl;

        throw std::exception();
    }

    __iterations = N;
    __dt = 1.0f / float(__iterations);
}

int FlowPropagatorPayload::getIterations() const {
    return __iterations;
}

float FlowPropagatorPayload::getDt() const {
    return __dt;
}


void FlowPropagatorPayload::setBorder(const int border) {

    if(border < 0) {
        std::cerr << "ERROR: FlowPropagatorPayload::setBorder(): border should be greater of equal zero: "
            << border << std::endl;

        throw std::exception();
    }

    __border = border;
}


int FlowPropagatorPayload::getBorder() const {
    return __border;
}

//#########################
// Stage inputs
//#########################
void FlowPropagatorPayload::setInputFlow(GPUImage inputFlow) {

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

void FlowPropagatorPayload::setScalarPayload(GPUImage scalarPayload) {

    if(scalarPayload.depth() != 1) {
        std::cerr << "ERROR: FlowPropagatorPayload::setscalarPayload(): input flow should have depth 2: "
            << scalarPayload.depth() << std::endl;
        throw std::exception();
    }

    if(scalarPayload.itemSize() != 4) {
        std::cerr << "ERROR: FlowPropagatorPayload::setscalarPayload(): input flow should have item size 4: "
            << scalarPayload.itemSize() << std::endl;
        throw std::exception();
    }

    // check size with respect to __inputFlow
    if(scalarPayload.height() != __inputFlow.height() ||
        scalarPayload.width() != __inputFlow.width()) {
        std::cerr << "ERROR: FlowPropagatorPayload::setscalarPayload(): scalar field shape" <<
            "does not match with input flow" << std::endl;
        throw std::exception();
    }

    __inputScalar = scalarPayload;
    __scalarPayloadSet = true;
}

void FlowPropagatorPayload::setVectorPayload(GPUImage vectorPayload) {

    if(vectorPayload.depth() != 2) {
        std::cerr << "ERROR: FlowPropagatorPayload::setvectorPayload(): input flow should have depth 2: "
            << vectorPayload.depth() << std::endl;
        throw std::exception();
    }

    if(vectorPayload.itemSize() != 4) {
        std::cerr << "ERROR: FlowPropagatorPayload::setvectorPayload(): input flow should have item size 4: "
            << vectorPayload.itemSize() << std::endl;
        throw std::exception();
    }

    // check size with respect to __inputFlow
    if(vectorPayload.height() != __inputFlow.height() ||
        vectorPayload.width() != __inputFlow.width()) {
        std::cerr << "ERROR: FlowPropagatorPayload::setvectorPayload(): scalar field shape" <<
            "does not match with input flow" << std::endl;
        throw std::exception();
    }

    __inputVector = vectorPayload;
    __vectorPayloadSet = true;
}


//#########################
// Stage outputs
//#########################
GPUImage FlowPropagatorPayload::getPropagatedFlow() {
    return __propagatedFlow_Y;
}

GPUImage FlowPropagatorPayload::getPropagatedScalar() {
    return __propagatedScalar_Y;
}

GPUImage FlowPropagatorPayload::getPropagatedVector() {
    return __propagatedVector_Y;
}


//###############################################
// LaxWendroffPropagator
//###############################################

LaxWendroffPropagator::LaxWendroffPropagator() {
    __iterations = 0;
    __dt = 0.0f;

    __inputFlowSet = false;
    __inputImageSet = false;
    __configured = false;
}

LaxWendroffPropagator::LaxWendroffPropagator(GPUImage inputFlow,
        GPUImage inputImage) :
    LaxWendroffPropagator() {

    setInputFlow(inputFlow);
    setInputImage(inputImage);
    configure();
}


LaxWendroffPropagator::~LaxWendroffPropagator() {
    // nothing to do
}


void LaxWendroffPropagator::configure() {

    if(!__inputFlowSet) {
        std::cerr << "ERROR: LaxWendroffPropagator::configure(): input flow has not been set" << std::endl;
        throw std::exception();
    }

    if(!__inputImageSet) {
        std::cerr << "ERROR: LaxWendroffPropagator::configure(): input image has not been set" << std::endl;
        throw std::exception();
    }

    int height = __inputFlow.height();
    int width = __inputFlow.width();

    //##################
    // flow
    //##################
    __inputFlowTexture = GPUTexture(__inputFlow, cudaChannelFormatKindFloat);

    //##################
    // input image
    //##################
    __inputImageTexture = GPUTexture(__inputImage, cudaChannelFormatKindFloat);

    __propagatedImage_X = GPUImage(height, width, __inputImage.depth(), sizeof(float));
    __propagatedImageTexture_X = GPUTexture(__propagatedImage_X, cudaChannelFormatKindFloat);

    __propagatedImage_Y = GPUImage(height, width, __inputImage.depth(), sizeof(float));
    __propagatedImageTexture_Y = GPUTexture(__propagatedImage_Y, cudaChannelFormatKindFloat);

    // configure block and grid sizes
    __block = dim3(32, 32, 1);
    configureKernelGrid(height, width, __block, __grid);

    __configured = true;
}


void LaxWendroffPropagator::compute() {

    startTiming();

    if(!__configured) {
        std::cerr << "ERROR: LaxWendroffPropagator::compute(): stage not configured" << std::endl;
        throw std::exception();
    }

    if(__inputImage.depth() == 1) {
        // first iteration
        LaxWendroffY_k<float><<<__grid, __block, 0, __stream>>>(
            __inputFlowTexture.getTextureObject(),
            __inputImageTexture.getTextureObject(),
            __propagatedImage_Y.wrap<float>(),
            __dt);

        LaxWendroffX_k<float><<<__grid, __block, 0, __stream>>>(
            __inputFlowTexture.getTextureObject(),
            __propagatedImageTexture_Y.getTextureObject(),
            __propagatedImage_X.wrap<float>(),
            __dt);

        // remaining iterations
        for(int k = 0; k < __iterations -1; k ++) {
            LaxWendroffY_k<float><<<__grid, __block, 0, __stream>>>(
                __inputFlowTexture.getTextureObject(),
                __propagatedImageTexture_X.getTextureObject(),
                __propagatedImage_Y.wrap<float>(),
                __dt);

            LaxWendroffX_k<float><<<__grid, __block, 0, __stream>>>(
                __inputFlowTexture.getTextureObject(),
                __propagatedImageTexture_Y.getTextureObject(),
                __propagatedImage_X.wrap<float>(),
                __dt);
        }

    } else if(__inputImage.depth() == 4) {

        // first iteration
        LaxWendroffY_k<float4><<<__grid, __block, 0, __stream>>>(
            __inputFlowTexture.getTextureObject(),
            __inputImageTexture.getTextureObject(),
            __propagatedImage_Y.wrap<float4>(),
            __dt);

        LaxWendroffX_k<float4><<<__grid, __block, 0, __stream>>>(
            __inputFlowTexture.getTextureObject(),
            __propagatedImageTexture_Y.getTextureObject(),
            __propagatedImage_X.wrap<float4>(),
            __dt);

        // remaining iterations
        for(int k = 0; k < __iterations -1; k ++) {
            LaxWendroffY_k<float4><<<__grid, __block, 0, __stream>>>(
                __inputFlowTexture.getTextureObject(),
                __propagatedImageTexture_X.getTextureObject(),
                __propagatedImage_Y.wrap<float4>(),
                __dt);

            LaxWendroffX_k<float4><<<__grid, __block, 0, __stream>>>(
                __inputFlowTexture.getTextureObject(),
                __propagatedImageTexture_Y.getTextureObject(),
                __propagatedImage_X.wrap<float4>(),
                __dt);
        }
    }

    

    stopTiming();
}


void LaxWendroffPropagator::setIterations(const int N) {

    if(N <= 0) {
        std::cerr << "ERROR: LaxWendroffPropagator::setIterations(): iterations less than zero: "
            << N << std::endl;
        throw std::exception();
    }

    __iterations = N;
    __dt = 1.0f / float(__iterations);
}


int LaxWendroffPropagator::getIterations() const {
    return __iterations;
}


float LaxWendroffPropagator::getDt() const {
    return __dt;
}



void LaxWendroffPropagator::setInputFlow(GPUImage inputFlow) {

    if(inputFlow.depth() != 2) {
        std::cerr << "ERROR: LaxWendroffPropagator::setInputFlow(): input flow should have depth 2: "
            << inputFlow.depth() << std::endl;
        throw std::exception();
    }

    if(inputFlow.itemSize() != 4) {
        std::cerr << "ERROR: LaxWendroffPropagator::setInputFlow(): input flow should have item size 4: "
            << inputFlow.itemSize() << std::endl;
        throw std::exception();
    }

    __inputFlow = inputFlow;
    __inputFlowSet = true;
}


void LaxWendroffPropagator::setInputImage(GPUImage img) {

    if(img.depth() != 1 && img.depth() != 4) {
        std::cerr << "ERROR: LaxWendroffPropagator::setInputImage(): input flow should have depth 1 or 4, got: "
            << img.depth() << std::endl;
        throw std::exception();
    }

    if(img.itemSize() != 4) {
        std::cerr << "ERROR: LaxWendroffPropagator::setInputImage(): input image should have item size 4: "
            << img.itemSize() << std::endl;
        throw std::exception();
    }

    // check size with respect to __inputFlow
    if(img.height() != __inputFlow.height() ||
        img.width() != __inputFlow.width()) {
        std::cerr << "ERROR: LaxWendroffPropagator::setInputImage(): image shape" <<
            "does not match with input flow" << std::endl;
        throw std::exception();
    }

    __inputImage = img;
    __inputImageSet = true;
}


GPUImage LaxWendroffPropagator::getFlow() {
    return __inputFlow;
}


GPUImage LaxWendroffPropagator::getPropagatedImage() {
    return __propagatedImage_X;
}



}; // namespace gpu
}; // namespace flowfilter
