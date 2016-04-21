/**
 * \file display_k.h
 * \brief Kernel declarations to convert optical flow fields to color representation.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_DISPLAY_K_H_
#define FLOWFILTER_GPU_DISPLAY_K_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/gpu/image.h"


namespace flowfilter {
namespace gpu {


__global__ void flowToColor_k(gpuimage_t<float2> inputFlow,
                              cudaTextureObject_t colorWheel,
                              const float maxflow,
                              gpuimage_t<uchar4> colorFlow);


}; // namepsace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_DISPLAY_K_H_