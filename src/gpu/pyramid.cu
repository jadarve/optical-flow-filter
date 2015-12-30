/**
 * \file pyramid.h
 * \brief Classes for computing image pyramids.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#include <iostream>
#include <exception>

#include "flowfilter/gpu/util.h"
#include "flowfilter/gpu/error.h"
#include "flowfilter/gpu/pyramid.h"
#include "flowfilter/gpu/device/pyramid_k.h"


namespace flowfilter {
namespace gpu {


ImagePyramid::ImagePyramid() :
    Stage() {

    __configured = false;
    __inputImageSet = false;
    __levels = 0;

}

ImagePyramid::ImagePyramid(flowfilter::gpu::GPUImage image,
    const int levels) :
    Stage() {

    __configured = false;
    __inputImageSet = false;

    setLevels(levels);
    setInputImage(image);
    configure();
}

ImagePyramid::~ImagePyramid() {

    // nothing to do
}


void ImagePyramid::configure() {

    if(!__inputImageSet) {
        std::cerr << "ERROR: ImageModel::configure(): input image has not been set" << std::endl;
        throw std::exception();
    }

    int height = __inputImage.height();
    int width = __inputImage.width();
    bool isUchar8 = __inputImage.itemSize() == sizeof(unsigned char);
    __block = dim3(32, 32, 1);

    // input image texture
    if(isUchar8) {
        // wraps __inputImage with normalized texture
        __inputImageTexture = GPUTexture(__inputImage, cudaChannelFormatKindUnsigned, cudaReadModeNormalizedFloat);
    } else {
        // wraps __inputImage with float texture
        __inputImageTexture = GPUTexture(__inputImage, cudaChannelFormatKindFloat);
    }

    // for levels 0 to H - 2
    for(int h = 0; h < __levels -1; h ++) {

        // downsampling in X
        width /= 2;
        GPUImage img_x(height, width, 1, __inputImage.itemSize());
        __pyramidX.push_back(img_x);
        
        dim3 gx(0, 0, 0);
        configureKernelGrid(height, width, __block, gx);
        __gridX.push_back(gx);

        // downsampling in Y
        height /= 2;
        GPUImage img_y(height, width, 1, __inputImage.itemSize());
        __pyramidY.push_back(img_y);

        dim3 gy(0, 0, 0);
        configureKernelGrid(height, width, __block, gy);
        __gridY.push_back(gy);

        // configure textures
        if(isUchar8) {
            GPUTexture tex_x(img_x, cudaChannelFormatKindUnsigned, cudaReadModeNormalizedFloat);
            GPUTexture tex_y(img_y, cudaChannelFormatKindUnsigned, cudaReadModeNormalizedFloat);

            __pyramidTextureX.push_back(tex_x);
            __pyramidTextureY.push_back(tex_y);
        } else {
            GPUTexture tex_x(img_x, cudaChannelFormatKindFloat);
            GPUTexture tex_y(img_y, cudaChannelFormatKindFloat);

            __pyramidTextureX.push_back(tex_x);
            __pyramidTextureY.push_back(tex_y);
        }
    }

    __configured = true;
}

void ImagePyramid::compute() {

    startTiming();

    if(!__configured) {
        std::cerr << "ERROR: ImagePyramid::compute(): stage not configured" << std::endl;
        throw std::exception();
    }

    bool isUchar8 = __inputImage.itemSize() == sizeof(unsigned char);

    if(isUchar8) {

        for(int h = 0; h < __levels -1; h ++) {

            // downsample in X
            if(h == 0) {
                imageDownX_uint8_k<<<__gridX[h], __block, 0, __stream>>>(
                    __inputImageTexture.getTextureObject(),
                    __pyramidX[h].wrap<unsigned char>());

            } else {
                imageDownX_uint8_k<<<__gridX[h], __block, 0, __stream>>>(
                    __pyramidTextureY[h-1].getTextureObject(),
                    __pyramidX[h].wrap<unsigned char>());
            }

            // downsample in Y
            imageDownY_uint8_k<<<__gridY[h], __block, 0, __stream>>>(
                __pyramidTextureX[h].getTextureObject(),
                __pyramidY[h].wrap<unsigned char>());
        }

    } else {

        for(int h = 0; h < __levels -1; h ++) {

            // downsample in X
            if(h == 0) {
                imageDownX_float_k<<<__gridX[h], __block, 0, __stream>>>(
                    __inputImageTexture.getTextureObject(),
                    __pyramidX[h].wrap<float>());

            } else {
                imageDownX_float_k<<<__gridX[h], __block, 0, __stream>>>(
                    __pyramidTextureY[h-1].getTextureObject(),
                    __pyramidX[h].wrap<float>());
            }

            // downsample in Y
            imageDownY_float_k<<<__gridY[h], __block, 0, __stream>>>(
                __pyramidTextureX[h].getTextureObject(),
                __pyramidY[h].wrap<float>());
        }
    }

    stopTiming();
}


//#########################
// Stage inputs
//#########################
void ImagePyramid::setInputImage(flowfilter::gpu::GPUImage img) {

    // check if image is a gray scale image with pixels 1 byte long
    if(img.depth() != 1) {
        std::cerr << "ERROR: ImagePyramid::setInputImage(): image depth should be 1: " << img.depth() << std::endl;
        throw std::exception();
    }

    if(img.itemSize() != sizeof(unsigned char) &&
        img.itemSize() != sizeof(float)) {

        std::cerr << "ERROR: ImagePyramid::setInputImage(): item size should be 1 or 4: " << img.itemSize() << std::endl;
        throw std::exception();
    }

    __inputImage = img;
    __inputImageSet = true;
}

void ImagePyramid::setLevels(const int levels) {

    if(levels <= 0) {
        std::cerr << "ERROR: ImagePyramid::setLevels(): " <<
            "levels should be greater than zero: " << levels << std::endl;
        throw std::exception();
    }

    __levels = levels;
}


//#########################
// Stage outputs
//#########################
flowfilter::gpu::GPUImage ImagePyramid::getImage(int level) {

    if(level < 0 || level >= __levels){
        std::cerr << "ERROR: ImagePyramid::getImage(): level index out of range: " << level << std::endl;
        throw std::exception();
    }

    if(level == 0) {
        return __inputImage;
    } else {
        return __pyramidY[level-1];
    }
}


int ImagePyramid::getLevels() const {
    return __levels;
}



}; // namespace gpu
}; // namespace flowfilter
