/**
 * \file update_k.h
 * \brief Kernel declarations for optical flow update computation.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_UPDATE_K_H_
#define FLOWFILTER_GPU_UPDATE_K_H_


#include "flowfilter/gpu/image.h"


namespace flowfilter {
namespace gpu {

__global__ void flowUpdate_k(gpuimage_t<float> newImage,
                             gpuimage_t<float2> newImageGradient,
                             gpuimage_t<float> oldImage,
                             gpuimage_t<float2> oldFlow,
                             gpuimage_t<float> imageUpdated,
                             gpuimage_t<float2> flowUpdated,
                             const float gamma, const float maxflow);


__global__ void deltaFlowUpdate_k(gpuimage_t<float> newImage,
                                  gpuimage_t<float2> newImageGradient,
                                  gpuimage_t<float> oldImage,
                                  gpuimage_t<float2> oldDeltaFlow,
                                  cudaTextureObject_t oldFlowTexture,
                                  gpuimage_t<float> imageUpdated,
                                  gpuimage_t<float2> deltaFlowUpdated,
                                  gpuimage_t<float2> flowUpdated,
                                  const float gamma, const float maxflow);
}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_UPDATE_K_H_