/**
 * \file camera_k.h
 * \brief Kernel methods for camera functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */
#ifndef FLOWFILTER_GPU_CAMERA_K_H_
#define FLOWFILTER_GPU_CAMERA_K_H_

#include "flowfilter/gpu/camera.h"

namespace flowfilter {
namespace gpu {


/**
 * \brief returns 3D coordinate in image plane for corresponding pixel coordinate
 *
 * \param cam perspective camera
 * \param pix pixel coordinate in (col, row) order.
 */
inline __device__ float3 pixelToCameraCoordinates(const perspectiveCamera& cam, const int2& pix) {

    return make_float3( (pix.x - cam.centerX)/cam.alphaX,
                        (pix.y - cam.centerY)/cam.alphaY,
                        1);
}

} // namespace gpu
} // namespace flowfilter

#endif /* FLOWFILTER_GPU_CAMERA_K_H_ */