/**
 * \file rotation.cu
 * \brief Classes for working with rotational optical flow fields.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <exception>

#include "flowfilter/gpu/util.h"
 
#include "flowfilter/gpu/rotation.h"

#include "flowfilter/gpu/device/rotation_k.h"


namespace flowfilter {
namespace gpu {


RotationalFlowImagePredictor::RotationalFlowImagePredictor() {

    __configured = false;
    __inputImageSet = false;
}


RotationalFlowImagePredictor::RotationalFlowImagePredictor(perspectiveCamera cam) :
    RotationalFlowImagePredictor() {

    setCamera(cam);
}


RotationalFlowImagePredictor::RotationalFlowImagePredictor(perspectiveCamera cam,
        flowfilter::gpu::GPUImage inputImage) :
    RotationalFlowImagePredictor(cam) {

    setInputImage(inputImage);
    configure();
}


RotationalFlowImagePredictor::~RotationalFlowImagePredictor() {
    // nothing to do
}


void RotationalFlowImagePredictor::configure() {

    if(!__inputImageSet) {
        std::cerr << "RotationalFlowImagePredictor::configure(): input image not set" << std::endl;
        throw std::exception();
    }

    int height = __inputImage.height();
    int width = __inputImage.width();

    __opticalFlow = GPUImage(height, width, 2, sizeof(float));
    __propagator = LaxWendroffPropagator(__opticalFlow, __inputImage);

    // configure block and grid sizes
    __block = dim3(32, 32, 1);
    configureKernelGrid(height, width, __block, __grid);

    __configured = true;
}


void RotationalFlowImagePredictor::compute() {

    startTiming();

    if(!__configured) {
        std::cerr << "RotationalFlowImagePredictor::compute(): stage not configured" << std::endl;
        throw std::exception();
    }

    // compute optical flow
    rotationalOpticalFlow_k<<<__grid, __block, 0, __stream>>>(__camera,
        __angularVelocity, __opticalFlow.wrap<float2>());

    // compute image prediction
    __propagator.compute();

    stopTiming();
}


void RotationalFlowImagePredictor::setInputImage(GPUImage inputImage) {

    // TODO: validate

    __inputImage = inputImage;
    __inputImageSet = true;
}


GPUImage RotationalFlowImagePredictor::getPredictedImage() {

    return __propagator.getPropagatedImage();
}

GPUImage RotationalFlowImagePredictor::getOpticalFlow() {

    return __opticalFlow;
}


void RotationalFlowImagePredictor::setCamera(perspectiveCamera cam) {
    __camera = cam;
}


void RotationalFlowImagePredictor::setAngularVelocity(const float wx, const float wy, const float wz) {

    __angularVelocity.x = wx;
    __angularVelocity.y = wy;
    __angularVelocity.z = wz;
}


void RotationalFlowImagePredictor::setIterations(const int iterations) {

    __propagator.setIterations(iterations);
}

int RotationalFlowImagePredictor::getIterations() const {

    return __propagator.getIterations();
}


} // namespace gpu
} // namespace flowfilter
