/**
 * \file flowfilter.cu
 * \brief Optical flow filter classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#include <iostream>
#include <string>
#include <exception>
#include <stdexcept>
#include <cmath>

#include "flowfilter/gpu/util.h"
#include "flowfilter/gpu/error.h"
#include "flowfilter/gpu/flowfilter.h"

namespace flowfilter {
namespace gpu {


FlowFilter::FlowFilter() :
    Stage() {

    __height = 0;
    __width = 0;
    __configured = false;
    __inputImageSet = false;
}

FlowFilter::FlowFilter(flowfilter::gpu::GPUImage inputImage) :
    Stage() {


    setInputImage(inputImage);
    configure();
}

FlowFilter::FlowFilter(const int height, const int width) :
    FlowFilter(height, width, 1, 1.0, 1.0) {
        
}

FlowFilter::FlowFilter(const int height, const int width,
        const int smoothIterations,
        const float maxflow,
        const float gamma) :
    Stage() {

    if(height <= 0) {
        std::cerr << "ERROR: FlowFilter::FlowFilter(): height should be greater than zero: " << height << std::endl;
        throw std::invalid_argument("FlowFilter::FlowFilter(): height should be greater than zero, got: " + std::to_string(height));
    }

    if(width <= 0) {
        std::cerr << "ERROR: FlowFilter::FlowFilter(): width should be greater than zero: " << width << std::endl;
        throw std::invalid_argument("FlowFilter::FlowFilter(): width should be greater than zero, got: " + std::to_string(width));
    }

    // __height = height;
    // __width = width;
    __height = 0;
    __width = 0;
    __configured = false;
    __inputImageSet = false;

    // creates a GPUImage for storing input image internally
    GPUImage inputImage = GPUImage(height, width, 1, sizeof(unsigned char));

    setInputImage(inputImage);
    configure();
    setGamma(gamma);
    setMaxFlow(maxflow);
    setSmoothIterations(smoothIterations);
}


FlowFilter::~FlowFilter() {
    // nothing to do
}


void FlowFilter::configure() {

    if(!__inputImageSet) {
        std::cerr << "ERROR: FlowFilter::configure(): input image has not been set" << std::endl;
        throw std::logic_error("FlowFilter::configure(): input image has not been set");
    }

    // connect the blocks
    __imageModel = ImageModel(__inputImage);

    // dummy flow field use to instanciate the update block
    // This is necessary to break the circular dependency
    // between propagation and update blocks.
    GPUImage dummyFlow(__height, __width, 2, sizeof(float));

    // FIXME: find good default values
    __update = FlowUpdate(dummyFlow,
        __imageModel.getImageConstant(),
        __imageModel.getImageGradient(),
        1.0, 1.0);

    __smoother = FlowSmoother(__update.getUpdatedFlow(), 1);

    __propagator = FlowPropagator(__smoother.getSmoothedFlow(), 1);

    // set the input flow of the update block to the output
    // of the propagator. This replaces dummyFlow previously
    // assigned to the update
    __update.setInputFlow(__propagator.getPropagatedFlow());
    
    // clear buffers
    __propagator.getPropagatedFlow().clear();
    __update.getUpdatedFlow().clear();
    __update.getUpdatedImage().clear();
    __smoother.getSmoothedFlow().clear();

    __configured = true;
    __firstLoad = true;
}


void FlowFilter::compute() {

    startTiming();

    // compute image model
    __imageModel.compute();

    if(__firstLoad) {
        std::cout << "FlowFilter::compute(): fisrt load" << std::endl;

        // set the old image value to current
        // computed constant brightness parameter
        GPUImage imConstant = __imageModel.getImageConstant();
        __update.getUpdatedImage().copyFrom(imConstant);
        
        __firstLoad = false;
    }

    // propagate old flow
    __propagator.compute();

    // update
    __update.compute();
    
    // smooth updated flow
    __smoother.compute();

    stopTiming();
}

void FlowFilter::computeImageModel() {

    startTiming();

    __imageModel.compute();

    stopTiming();
}


void FlowFilter::computePropagation() {

    startTiming();

    __propagator.compute();

    stopTiming();
}


void FlowFilter::computeUpdate() {

    startTiming();

    if(__firstLoad) {
        std::cout << "FlowFilter::compute(): fisrt load" << std::endl;

        // set the old image value to current
        // computed constant brightness parameter
        GPUImage imConstant = __imageModel.getImageConstant();
        __update.getUpdatedImage().copyFrom(imConstant);
        
        __firstLoad = false;
    }

    // update
    __update.compute();
    
    // smooth updated flow
    __smoother.compute();

    stopTiming();
}


void FlowFilter::setInputImage(GPUImage inputImage) {

    if(inputImage.depth() != 1) {
        std::cerr << "ERROR: FlowFilter::setInputImage(): input image should have depth 1: " << inputImage.depth() << std::endl;
        throw std::invalid_argument("FlowFilter::setInputImage(): input image should have depth 1, got: " + std::to_string(inputImage.depth()));
    }

    if(inputImage.itemSize() != sizeof(unsigned char) && inputImage.itemSize() != sizeof(float)) {
        std::cerr << "ERROR: FlowFilter::setInputImage(): item size should be 1 or 4: " << inputImage.itemSize() << std::endl;
        throw std::invalid_argument("FlowFilter::setInputImage(): item size should be 1 or 4: " + std::to_string(inputImage.itemSize()));
    }

    __inputImage = inputImage;
    __height = __inputImage.height();
    __width = __inputImage.width();
    __inputImageSet = true;
}

void FlowFilter::loadImage(flowfilter::image_t& image) {

    __inputImage.upload(image);

    // if(__firstLoad) {

    //     std::cout << "FlowFilter::loadImage(): fisrt load" << std::endl;

    //     // compute image model parameters
    //     __imageModel.compute();

    //     // set the old image value to current
    //     // computed constant brightness parameter
    //     GPUImage imConstant = __imageModel.getImageConstant();
    //     __update.getUpdatedImage().copyFrom(imConstant);
        
    //     __firstLoad = false;
    // }
}

void FlowFilter::downloadFlow(flowfilter::image_t& flow) {
    __smoother.getSmoothedFlow().download(flow);
}

void FlowFilter::downloadImage(flowfilter::image_t& image) {
    __update.getUpdatedImage().download(image);
}

// void FlowFilter::downloadImageGradient(flowfilter::image_t& gradient) {
//     __imageModel.getImageGradient().download(gradient);
// }

// void FlowFilter::downloadImageConstant(flowfilter::image_t& image) {
//     __imageModel.getImageConstant().download(image);
// }

// void FlowFilter::downloadImageUpdated(flowfilter::image_t& image) {
//     __update.getUpdatedImage().download(image);
// }

// void FlowFilter::downloadFlowUpdated(flowfilter::image_t& flow) {
//     __update.getUpdatedFlow().download(flow);
// }

// void FlowFilter::downloadSmoothedFlow(flowfilter::image_t& flow) {
//     __smoother.getSmoothedFlow().download(flow);
// }

GPUImage FlowFilter::getFlow() {
    return __update.getUpdatedFlow();
}


float FlowFilter::getGamma() const {
    return __update.getGamma();
}


void FlowFilter::setGamma(const float gamma) {

    // scale gamma if input image is uint8
    if(__inputImage.itemSize() == 1){
        __update.setGamma(gamma / (255.0f*255.0f));
    } else {
        __update.setGamma(gamma);
    }
    
}


float FlowFilter::getMaxFlow() const {
    return __update.getMaxFlow();
}


void FlowFilter::setMaxFlow(const float maxflow) {
    __update.setMaxFlow(maxflow);
    __propagator.setIterations(int(ceilf(maxflow)));
}


int FlowFilter::getSmoothIterations() const {
    return __smoother.getIterations();
}


void FlowFilter::setSmoothIterations(const int N) {
    __smoother.setIterations(N);
}


void FlowFilter::setPropagationBorder(const int border) {
    __propagator.setBorder(border);
}


int FlowFilter::getPropagationBorder() const {
    return __propagator.getBorder();
}


int FlowFilter::getPropagationIterations() const {
    return __propagator.getIterations();
}

int FlowFilter::height() const {
    return __height;
    
}

int FlowFilter::width() const {
    return __width;
}



//###############################################
// DeltaFlowFilter
//###############################################

DeltaFlowFilter::DeltaFlowFilter() : 
    Stage() {

    __configured = false;
    __firstLoad = true;

    __inputImageSet = false;
    __inputFlowSet = false;

}

DeltaFlowFilter::DeltaFlowFilter(flowfilter::gpu::GPUImage inputImage,
    flowfilter::gpu::GPUImage inputFlow) :
    Stage() {

    __configured = false;
    __firstLoad = true;

    __inputImageSet = false;
    __inputFlowSet = false;

    setInputImage(inputImage);
    setInputFlow(inputFlow);
    configure();
}


DeltaFlowFilter::~DeltaFlowFilter() {

    // nothing to do
}


void DeltaFlowFilter::configure() {

    if(!__inputFlowSet) {
        std::cerr << "ERROR: DeltaFlowFilter::configure(): input flow not set" << std::endl;
        throw std::exception();
    }

    if(!__inputImageSet) {
        std::cerr << "ERROR: DeltaFlowFilter::configure(): input image not set" << std::endl;
        throw std::exception();
    }

    int height = __inputImage.height();
    int width = __inputImage.width();

    __imageModel = ImageModel(__inputImage);

    // dummy inputs to create delta flow update
    GPUImage dummyDeltaFlow(height, width, 2, sizeof(float));
    GPUImage dummyImageOld(height, width, 1, sizeof(float));

    // create delta flow update stage
    __update = DeltaFlowUpdate(__inputFlow, dummyDeltaFlow,
        dummyImageOld, __imageModel.getImageConstant(),
        __imageModel.getImageGradient());

    // flow smoother
    __smoother = FlowSmoother(__update.getUpdatedFlow(), 1);

    // propagator with payload
    __propagator = FlowPropagatorPayload(__smoother.getSmoothedFlow(),
        __update.getUpdatedImage(), __update.getUpdatedDeltaFlow());


    // replace dummy inputs with propagated outputs
    __update.setInputDeltaFlow(__propagator.getPropagatedVector());
    __update.setInputImageOld(__propagator.getPropagatedScalar());


    // clear buffers
    __imageModel.getImageConstant().clear();
    __imageModel.getImageGradient().clear();

    __propagator.getPropagatedFlow().clear();
    __propagator.getPropagatedScalar().clear();
    __propagator.getPropagatedVector().clear();

    __update.getUpdatedFlow().clear();
    __update.getUpdatedDeltaFlow().clear();
    __update.getUpdatedImage().clear();

    __smoother.getSmoothedFlow().clear();

    __configured = true;
    __firstLoad = true;
}


void DeltaFlowFilter::compute() {

    startTiming();

    // compute image model
    __imageModel.compute();

    if(__firstLoad) {
        std::cout << "DeltaFlowFilter::compute(): fisrt load" << std::endl;

        // set the old image value to current
        // computed constant brightness parameter
        GPUImage imConstant = __imageModel.getImageConstant();
        __update.getUpdatedImage().copyFrom(imConstant);
        
        __firstLoad = false;
    }

    // propagate old flow
    __propagator.compute();

    // update
    __update.compute();
    
    // smooth updated flow
    __smoother.compute();

    stopTiming();
}


void DeltaFlowFilter::computeImageModel() {

    startTiming();

    __imageModel.compute();

    stopTiming();
}


void DeltaFlowFilter::computePropagation() {

    startTiming();

    __propagator.compute();

    stopTiming();
}


void DeltaFlowFilter::computeUpdate() {

    startTiming();

    if(__firstLoad) {
        std::cout << "DeltaFlowFilter::compute(): fisrt load" << std::endl;

        // set the old image value to current
        // computed constant brightness parameter
        GPUImage imConstant = __imageModel.getImageConstant();
        __update.getUpdatedImage().copyFrom(imConstant);
        __propagator.getPropagatedScalar().copyFrom(imConstant);
        
        __firstLoad = false;
    }

    // update
    __update.compute();
    
    // smooth updated flow
    __smoother.compute();

    stopTiming();
}


void DeltaFlowFilter::setInputImage(GPUImage inputImage) {

    if(inputImage.depth() != 1) {
        std::cerr << "ERROR: DeltaFlowFilter::setInputImage(): input image should have depth 1: "
            << inputImage.depth() << std::endl;
        throw std::exception();
    }

    if(inputImage.itemSize() != sizeof(unsigned char) &&
        inputImage.itemSize() != sizeof(float)) {
        std::cerr << "ERROR: DeltaFlowFilter::setInputImage(): input image should have item size 4: "
            << inputImage.itemSize() << std::endl;
        throw std::exception();
    }

    __inputImage = inputImage;
    __inputImageSet = true;
}


void DeltaFlowFilter::setInputFlow(GPUImage inputFlow) {

    if(inputFlow.depth() != 2) {
        std::cerr << "ERROR: DeltaFlowFilter::setInputFlow(): input flow should have depth 2: "
            << inputFlow.depth() << std::endl;
        throw std::exception();
    }

    if(inputFlow.itemSize() != 4) {
        std::cerr << "ERROR: DeltaFlowFilter::setInputFlow(): input flow should have item size 4: "
            << inputFlow.itemSize() << std::endl;
        throw std::exception();
    }

    __inputFlow = inputFlow;
    __inputFlowSet = true;
}


GPUImage DeltaFlowFilter::getFlow() {
    return __smoother.getSmoothedFlow();
}

GPUImage DeltaFlowFilter::getImage() {
    return __update.getUpdatedImage();
}


float DeltaFlowFilter::getGamma() const {
    return __update.getGamma();
}


void DeltaFlowFilter::setGamma(const float gamma) {

    // scale gamma if input image is uint8
    if(__inputImage.itemSize() == 1) {
        __update.setGamma(gamma / (255.0f*255.0f));    
    } else {
        __update.setGamma(gamma);
    }
    
}


float DeltaFlowFilter::getMaxFlow() const {
    return __update.getMaxFlow();
}


void DeltaFlowFilter::setMaxFlow(const float maxflow) {
    __update.setMaxFlow(maxflow);
    __propagator.setIterations(int(ceilf(maxflow)));
}


int DeltaFlowFilter::getSmoothIterations() const {
    return __smoother.getIterations();
}


void DeltaFlowFilter::setSmoothIterations(const int N) {
    __smoother.setIterations(N);
}


void DeltaFlowFilter::setPropagationBorder(const int border) {
    __propagator.setBorder(border);
}


int DeltaFlowFilter::getPropagationBorder() const {
    return __propagator.getBorder();
}


int DeltaFlowFilter::getPropagationIterations() const {
    return __propagator.getIterations();
}


int DeltaFlowFilter::height() const {
    return __inputImage.height();
    
}


int DeltaFlowFilter::width() const {
    return __inputImage.width();
}


//###############################################
// PyramidalFlowFilter
//###############################################
PyramidalFlowFilter::PyramidalFlowFilter() : 
    Stage() {

    __height = 0;
    __width = 0;
    __levels = 0;
    __configured = false;
}


PyramidalFlowFilter::PyramidalFlowFilter(const int height, const int width, const int levels) :
    Stage() {


    __height = height;
    __width = width;
    __levels = levels;
    __configured = false;

    configure();
}


PyramidalFlowFilter::~PyramidalFlowFilter() {

    // nothing to do
}


void PyramidalFlowFilter::configure() {

    __inputImage = GPUImage(__height, __width, 1, sizeof(unsigned char));

    // image pyramid
    __imagePyramid = ImagePyramid(__inputImage, __levels);

    // top level filter block
    __topLevelFilter = FlowFilter(__imagePyramid.getImage(__levels -1));

    if(__levels > 1) {
        __lowLevelFilters.resize(__levels -1);

        GPUImage levelInputFlow = __topLevelFilter.getFlow();
        levelInputFlow.clear();

        for(int h = __levels -2; h >= 0; h --) {

            __lowLevelFilters[h] = DeltaFlowFilter(
                __imagePyramid.getImage(h), levelInputFlow);

            levelInputFlow = __lowLevelFilters[h].getFlow();
        }
    }

    // clear buffers
    __inputImage.clear();
    for(int h = 0; h < __levels; h ++) {
        __imagePyramid.getImage(h).clear();
    }

    __configured = true;
}


void PyramidalFlowFilter::compute() {

    startTiming();

    // compute image pyramid
    __imagePyramid.compute();

    if(__levels == 1) {
        __topLevelFilter.compute();

    } else {

        // compute image model and propagation for all levels

        __topLevelFilter.computeImageModel();
        __topLevelFilter.computePropagation();

        for(int h =0; h < __levels - 1; h ++) {
            __lowLevelFilters[h].computeImageModel();
            __lowLevelFilters[h].computePropagation();
        }

        // update
        __topLevelFilter.computeUpdate();

        for(int h =0; h < __levels - 1; h ++) {
            __lowLevelFilters[h].computeUpdate();
        }
    }

    stopTiming();
}

GPUImage PyramidalFlowFilter::getFlow() {

    if(__levels == 1) {
        return __topLevelFilter.getFlow();
    } else {
        return __lowLevelFilters[0].getFlow();
    }
}


void PyramidalFlowFilter::loadImage(image_t& image) {

    __inputImage.upload(image);
}


void PyramidalFlowFilter::downloadFlow(image_t& flow) {

    if(__levels == 1) {
        __topLevelFilter.downloadFlow(flow);
    } else {
        __lowLevelFilters[0].getFlow().download(flow);
    }
}


void PyramidalFlowFilter::downloadImage(image_t& image) {

    if(__levels == 1) {
        __topLevelFilter.downloadImage(image);
    } else {
        __lowLevelFilters[0].getImage().download(image);
    }
}


float PyramidalFlowFilter::getGamma(const int level) const {
    
    if(level < 0 || level >= __levels) {
        std::cerr << "ERROR: PyramidalFlowFilter::getGamma(): level index out of bounds: " << level << std::endl;
        throw std::exception();
    }

    if(level == __levels -1) {
        return __topLevelFilter.getGamma();
    } else {
        return __lowLevelFilters[level].getGamma();
    }
}


void PyramidalFlowFilter::setGamma(const int level, const float gamma) {

    if(level < 0 || level >= __levels) {
        std::cerr << "ERROR: PyramidalFlowFilter::setGamma(): level index out of bounds: " << level << std::endl;
        throw std::exception();
    }

    if(level == __levels -1) {
        __topLevelFilter.setGamma(gamma);
    } else {
        __lowLevelFilters[level].setGamma(gamma);
    }
}


void PyramidalFlowFilter::setGamma(const std::vector<float>& gamma) {

    if(gamma.size() != __levels) {
        std::cerr << "ERROR: PyramidalFlowFilter::setGamma(): gamma vector should be size " << __levels << ", got: " << gamma.size();
        throw std::exception();
    }

    for(int h = 0; h < __levels; h ++) {
        setGamma(h, gamma[h]);
    }
}


int PyramidalFlowFilter::getSmoothIterations(const int level) const {

    if(level < 0 || level >= __levels) {
        std::cerr << "ERROR: PyramidalFlowFilter::getSmoothIterations(): level index out of bounds: " << level << std::endl;
        throw std::exception();
    }

    if(level == __levels -1) {
        return __topLevelFilter.getSmoothIterations();
    } else {
        return __lowLevelFilters[level].getSmoothIterations();
    }
}


void PyramidalFlowFilter::setSmoothIterations(const int level, const int N) {

    if(level < 0 || level >= __levels) {
        std::cerr << "ERROR: PyramidalFlowFilter::setSmoothIterations(): level index out of bounds: " << level << std::endl;
        throw std::exception();
    }

    if(level == __levels -1) {
        __topLevelFilter.setSmoothIterations(N);
    } else {
        __lowLevelFilters[level].setSmoothIterations(N);
    }
}

void PyramidalFlowFilter::setSmoothIterations(const std::vector<int>& iterations) {

    if(iterations.size() != __levels) {
        std::cerr << "ERROR: PyramidalFlowFilter::setSmoothIterations(): iterations vector should be size " << __levels << ", got: " << iterations.size();
        throw std::exception();
    }

    for(int h = 0; h < __levels; h ++) {
        setSmoothIterations(h, iterations[h]);
    }
}


float PyramidalFlowFilter::getMaxFlow() const {

    if(__levels == 1) {
        return __topLevelFilter.getMaxFlow();
    } else {
        return __lowLevelFilters[0].getMaxFlow();
    }
}


void PyramidalFlowFilter::setMaxFlow(const float maxflow) {

    if(__levels == 1) {
        __topLevelFilter.setMaxFlow(maxflow);

    } else {

        float maxflowLevel = maxflow;

        for(int h = 0; h < __levels - 1; h ++) {

            __lowLevelFilters[h].setMaxFlow(maxflowLevel);
            maxflowLevel /= 2.0f;
        }

        __topLevelFilter.setMaxFlow(maxflowLevel);
    }
}


void PyramidalFlowFilter::setPropagationBorder(const int border) {
    __topLevelFilter.setPropagationBorder(border);

    if(__levels > 1) {
        for(int h = 0; h < __levels; h ++) {
            __lowLevelFilters[h].setPropagationBorder(border);
        }
    }
}


int PyramidalFlowFilter::getPropagationBorder() const {
    return __topLevelFilter.getPropagationBorder();
}


int PyramidalFlowFilter::height() const {
    return __height;
}


int PyramidalFlowFilter::width() const {
    return __width;
}


int PyramidalFlowFilter::levels() const {
    return __levels;
}


}; // namespace gpu
}; // namespace flowfilter