/**
 * \file misc_k.h
 * \brief Miscellaneous kernel declarations.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_MISC_K_H_
#define FLOWFILTER_GPU_MISC_K_H_


#include "flowfilter/gpu/image.h"


namespace flowfilter {
namespace gpu {

/**
 * \brief Multiplies an input vector field by a scalar constant.
 */
__global__ void scalarProductF2_k(gpuimage_t<float2> inputField,
                                  const float scalar,
                                  gpuimage_t<float2> outputField);

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_MISC_K_H_