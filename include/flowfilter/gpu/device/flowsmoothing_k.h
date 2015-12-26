/**
 * \file flowsmoothing_k.h
 * \brief Kernel declarations for image model computation.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_FLOWSMOOTHING_K_H_
#define FLOWFILTER_GPU_FLOWSMOOTHING_K_H_


#include "flowfilter/gpu/image.h"


namespace flowfilter {
namespace gpu {

__global__ void flowSmoothX_k(cudaTextureObject_t inputFlow,
                              gpuimage_t<float2> flowSmooth);


__global__ void flowSmoothY_k(cudaTextureObject_t inputFlow,
                              gpuimage_t<float2> flowSmooth);

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_FLOWSMOOTHING_K_H_