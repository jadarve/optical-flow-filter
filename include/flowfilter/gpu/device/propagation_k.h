/**
 * \file propagation_k.h
 * \brief Kernel declarations for flow propagation methods.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_PROPAGATION_K_H_
#define FLOWFILTER_GPU_PROPAGATION_K_H_


#include "flowfilter/gpu/device/image_k.h"


namespace flowfilter {
namespace gpu {

__global__ void flowPropagateX_k(cudaTextureObject_t inputFlow,
                                 gpuimage_t<float2> flowPropagated,
                                 const float dt, const int border);

__global__ void flowPropagateY_k(cudaTextureObject_t inputFlow,
                                 gpuimage_t<float2> flowPropagated,
                                 const float dt, const int border);


}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_PROPAGATION_K_H_