/**
 * \file pyramid_k.h
 * \brief Kernel declaration for computing image and flow pyramids
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_PYRAMID_K_H_
#define FLOWFILTER_GPU_PYRAMID_K_H_


#include "flowfilter/gpu/image.h"


namespace flowfilter {
namespace gpu {

__global__ void imageDownX_uint8_k(cudaTextureObject_t inputImage,
    gpuimage_t<unsigned char> imageDown);


__global__ void imageDownY_uint8_k(cudaTextureObject_t inputImage,
    gpuimage_t<unsigned char> imageDown);


__global__ void imageDownX_float_k(cudaTextureObject_t inputImage,
    gpuimage_t<float> imageDown);


__global__ void imageDownY_float_k(cudaTextureObject_t inputImage,
    gpuimage_t<float> imageDown);

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_PYRAMID_K_H_