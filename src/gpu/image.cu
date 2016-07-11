/**
 * \file image.cu
 * \brief type declarations for GPU image buffers.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <cstring>
#include <string>
#include <iostream>
#include <exception>
#include <stdexcept>

#include "flowfilter/gpu/image.h"
#include "flowfilter/gpu/gpu_deleter.h"
#include "flowfilter/gpu/error.h"

namespace flowfilter {
namespace gpu {

//#################################################
// GPUImage
//#################################################
GPUImage::GPUImage() {
    __width = 0;
    __height = 0;
    __depth = 0;
    __pitch = 0;
    __itemSize = 0;
}

GPUImage::GPUImage(const int height, const int width,
    const int depth, const int itemSize) {

    __height = height;
    __width = width;
    __depth = depth;
    __itemSize = itemSize;

    // allocate memory in GPU space
    allocate();
}

GPUImage::~GPUImage() {

    // nothing to do
    // device buffer is released by gpu_deleter
    // std::cout << "GPUImage::~GPUImage(): [" << 
    //     __height << ", " << __width << ", " << __depth << "] : " << __ptr_dev.use_count() << std::endl;
}

int GPUImage::height() const {
    return __height;
}

int GPUImage::width() const {
    return __width;
}

int GPUImage::depth() const {
    return __depth;
}

int GPUImage::pitch() const {
    return __pitch;
}

int GPUImage::itemSize() const {
    return __itemSize;
}

void* GPUImage::data() {
    return __ptr_dev.get();
}

std::shared_ptr<void> GPUImage::getDeviceBuffer() {
    return __ptr_dev;
}


void GPUImage::upload(flowfilter::image_t& img) {

    // check if device memory is allocated
    if(!__ptr_dev) {

        // set resolution to input image
        __width = img.width;
        __height = img.height;
        __depth = img.depth;
        __itemSize = img.itemSize;

        // allocate memory
        allocate();
    }

    // compare shapes
    if(compareShape(img)) {

        // print first 5 elements of img
        // for(int i = 0; i < 5; i ++) {
        //     std::cout << i << ": " << (int)static_cast<unsigned char*>(img.data)[i] << std::endl;
        // }

        // issue synchronous memory copy
        checkError(cudaMemcpy2D(__ptr_dev.get(), __pitch, img.data, img.pitch,
            __width*__depth*__itemSize, __height,
            cudaMemcpyHostToDevice));

        // TODO: add support for asynchronous copy

    } else {

        std::cerr << "ERROR: GPUImage::upload(): shapes do not match."
            << "required: [" << __height << ", " << __width << ", " << __depth << "][" << __itemSize << "], passed: "
            << "[" << img.height << ", " << img.width << ", " << img.depth << "][" << img.itemSize << "]" << std::endl;

        throw std::invalid_argument("GPUImage::upload(): shapes do not match. Required: [" +
            std::to_string(__height) + ", " + std::to_string(__width) + ", " + std::to_string(__depth) + "][" + std::to_string(__itemSize) + "], passed: [" +
            std::to_string(img.height) + ", " + std::to_string(img.width) + ", " + std::to_string(img.depth) + "][" + std::to_string(img.itemSize) + "]");
    }
}

void GPUImage::download(flowfilter::image_t& img) const {

    if(!__ptr_dev) {
        std::cerr << "ERROR: GPUImage::download(): unallocated image" << std::endl;
        return; // TODO: throw exception
    }

    if(compareShape(img)) {

        // issue synchronous memory copy
        checkError(cudaMemcpy2D(img.data, img.pitch, __ptr_dev.get(), __pitch,
            __width*__depth*__itemSize, __height, cudaMemcpyDeviceToHost));

        // print first 5 elements of img
        // for(int i = 0; i < 5; i ++) {
        //     std::cout << i << ": " << static_cast<float*>(img.data)[i] << std::endl;
        // }

    } else {
        std::cerr << "ERROR: GPUImage::download(): shapes do not match."
            << "required: [" << __height << ", " << __width << ", " << __depth << "][" << __itemSize << "], passed: "
            << "[" << img.height << ", " << img.width << ", " << img.depth << "][" << img.itemSize << "]" << std::endl;

        throw std::invalid_argument("GPUImage::download(): shapes do not match. Required: [" +
            std::to_string(__height) + ", " + std::to_string(__width) + ", " + std::to_string(__depth) + "][" + std::to_string(__itemSize) + "], passed: [" +
            std::to_string(img.height) + ", " + std::to_string(img.width) + ", " + std::to_string(img.depth) + "][" + std::to_string(img.itemSize) + "]");
    }
}

void GPUImage::copyFrom(GPUImage& img) {

    if(compareShapeGPU(img)) {

        // issue synchronous memory copy
        checkError(cudaMemcpy2D(__ptr_dev.get(), __pitch, 
            img.__ptr_dev.get(), img.__pitch,
            __width*__depth*__itemSize, __height, cudaMemcpyDeviceToDevice));

    } else {
        std::cerr << "ERROR: GPUImage::copyFrom(): shapes do not match."
            << "required: [" << __height << ", " << __width << ", " << __depth << "][" << __itemSize << "], passed: "
            << "[" << img.__height << ", " << img.__width << ", " << img.__depth << "][" << img.__itemSize << "]" << std::endl;

        throw std::invalid_argument("GPUImage::copyFrom(): shapes do not match. Required: [" +
            std::to_string(__height) + ", " + std::to_string(__width) + ", " + std::to_string(__depth) + "][" + std::to_string(__itemSize) + "], passed: [" +
            std::to_string(img.__height) + ", " + std::to_string(img.__width) + ", " + std::to_string(img.__depth) + "][" + std::to_string(img.__itemSize) + "]");
    }
}

void GPUImage::clear() {

    checkError(cudaMemset2D(__ptr_dev.get(), __pitch,
        0, __width*__depth*__itemSize, __height));
}


void GPUImage::allocate() {

    // std::cout << "GPUImage::allocate()" << std::endl;

    void* buffer_dev = nullptr;
    checkError(cudaMallocPitch(&buffer_dev, &__pitch,
        __width*__depth*__itemSize, __height));

    // create a new shared pointer
    __ptr_dev = std::shared_ptr<void> (buffer_dev, gpu_deleter<void>());

    // std::cout << "\tpitch: " << __pitch << std::endl;

    // if(err != cudaSuccess) {
    //     std::cerr << "ERROR: GPUImage device memory allocation: " << cudaGetErrorString(err) << std::endl;
    //     // TODO: throw exception?
    //     throw std::bad_alloc("GPUImage::allocate(): device memory allocation error: " + cudaGetErrorString(err));
    // }
}

bool GPUImage::compareShape(const flowfilter::image_t& img) const {

    return __height == img.height &&
        __width == img.width &&
        __depth == img.depth &&
        __itemSize == img.itemSize;
}

bool GPUImage::compareShapeGPU(const flowfilter::gpu::GPUImage& img) const {

    return __height == img.height() &&
        __width == img.width() &&
        __depth == img.depth() &&
        __pitch == img.pitch() &&
        __itemSize == img.itemSize();
}



//#################################################
// GPUTexture
//#################################################
GPUTexture::GPUTexture() {

    // texture object is not valid
    __validTexture = false;
    __refCounter = std::make_shared<int>(0);
}

GPUTexture::GPUTexture( GPUImage& img, cudaChannelFormatKind format) :
    GPUTexture(img, format, cudaAddressModeClamp, 
        cudaFilterModePoint, cudaReadModeElementType, false) {
}

GPUTexture::GPUTexture( GPUImage& img,
                        cudaChannelFormatKind format,
                        cudaTextureReadMode readMode) : 
    GPUTexture(img, format, cudaAddressModeClamp,
        cudaFilterModePoint, readMode, false) {

}

GPUTexture::GPUTexture( flowfilter::gpu::GPUImage& img,
                        cudaChannelFormatKind format,
                        cudaTextureReadMode readMode,
                        const bool normalizedCoords) : 
    GPUTexture(img, format, cudaAddressModeClamp, cudaFilterModePoint,
        readMode, normalizedCoords) {

}

GPUTexture::GPUTexture( GPUImage& img,
                        cudaChannelFormatKind format,
                        cudaTextureAddressMode addressMode,
                        cudaTextureFilterMode filterMode,
                        cudaTextureReadMode readMode,
                        const bool normalizedCoords) {

    __refCounter = std::make_shared<int>(0);

    // hold input image
    __image = img;

    // configure CUDA texture
    configure(format, addressMode, filterMode, readMode, normalizedCoords);
}

GPUTexture::~GPUTexture() {

    // std::cout << "GPUTexture::~GPUTexture(): " <<  __refCounter.use_count() << " : " << __texture << std::endl;


    // only attempts to destroy the texture if the creation
    // was successful
    if(__refCounter.use_count() == 1) {
        if(__validTexture) {
            checkError(cudaDestroyTextureObject(__texture));    
        }
    }

    // __image destructor is called automatically and
    // devide buffer is released only if it's not being
    // shared in any other part of the program.
}

cudaTextureObject_t GPUTexture::getTextureObject() {
    return __texture;
}

GPUImage GPUTexture::getImage() {
    return __image;
}

void GPUTexture::configure( cudaChannelFormatKind format,
                            cudaTextureAddressMode addressMode,
                            cudaTextureFilterMode filterMode,
                            cudaTextureReadMode readMode,
                            const bool normalizedCoords) {
    
    __validTexture = false;

    int channels = __image.depth();
    if(channels > 4) {
        std::cerr << "ERROR: GPUTexture::configure(): image channels greater than 4: " << channels << std::endl;
        throw std::invalid_argument("GPUTexture::configure(): image channels greater than 4, got: " + std::to_string(channels));
    }

    // bit width of element
    int bitWidth = 8 * __image.itemSize();

    // bit width of each channel
    int w1 = bitWidth;  // there is at least one channel
    int w2 = channels >= 2? bitWidth : 0;
    int w3 = channels >= 3? bitWidth : 0;
    int w4 = channels == 4? bitWidth : 0;

    // std::cout << "GPUTexture::configure(): channel width: [" << w1 << ", " << w2 << ", " << w3 << ", " << w4 << "]" << std::endl;
    // std::cout << "GPUTexture::configure(): [" << __image.height() << ", " << __image.width() << ", " << __image.depth() << "] size: " << __image.itemSize() << " pitch: " << __image.pitch() << std::endl;

    // channel descriptor
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(w1, w2, w3, w4, format);

    // texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = addressMode;
    texDesc.addressMode[1] = addressMode;
    texDesc.filterMode = filterMode;
    texDesc.readMode = readMode;
    texDesc.normalizedCoords = normalizedCoords;

    // texture buffer descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.desc = channelDesc;
    resDesc.res.pitch2D.devPtr = __image.data();
    resDesc.res.pitch2D.pitchInBytes = __image.pitch();
    resDesc.res.pitch2D.width = __image.width();
    resDesc.res.pitch2D.height = __image.height();

    // creates texture
    checkError(cudaCreateTextureObject(&__texture, &resDesc, &texDesc, NULL));
    __validTexture = true;

    // cudaError_t err = cudaCreateTextureObject(&__texture, &resDesc, &texDesc, NULL);

    // std::cout << "GPUTexture::configure(): texture ID: " << __texture << std::endl;
    // if(err == cudaSuccess) {
    //     __validTexture = true;
    // } else {

    //     std::cerr << "ERROR: GPUTexture::configure(): texture creation: "
    //         << cudaGetErrorString(err) << std::endl;

    //     __validTexture = false;
    // }
}

}; // namespace gpu
}; // namespace flowfilter
