/**
 * \file rotation_k.h
 * \brief Kernel methods generating rotational flow fields.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_ROTATION_K_H_
#define FLOWFILTER_GPU_ROTATION_K_H_

#include "flowfilter/gpu/image.h"
#include "flowfilter/gpu/camera.h"

namespace flowfilter {
namespace gpu {

/**
 * \brief generates rotational optical flow field from camera parameters and
 *  angular velocity.
 */
__global__ void rotationalOpticalFlow_k(perspectiveCamera cam, 
    float3 w, gpuimage_t<float2> flowField);

} // namespace gpu
} // namespace flowfilter

#endif /* FLOWFILTER_GPU_ROTATION_K_H_ */
