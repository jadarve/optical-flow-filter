/**
 * \file rotation_k.cu
 * \brief Kernel methods generating rotational flow fields.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include "flowfilter/gpu/device/camera_k.h"
#include "flowfilter/gpu/device/image_k.h"
#include "flowfilter/gpu/device/math_k.h"
#include "flowfilter/gpu/device/rotation_k.h"


namespace flowfilter {
namespace gpu {

__global__ void rotationalOpticalFlow_k(perspectiveCamera cam, 
    float3 w, gpuimage_t<float2> flowField) {


    const int height = flowField.height;
    const int width = flowField.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // float3 w = make_float3(yaw, pitch, roll);
    float3 p = pixelToCameraCoordinates(cam, pix);

    float3 wp_cross = cross(w, p);

    float flowX = cam.alphaX*wp_cross.x + (cam.centerX - pix.x)*wp_cross.z;
    float flowY = cam.alphaY*wp_cross.y + (cam.centerY - pix.y)*wp_cross.z;

    float2 flow = make_float2(flowX, flowY);

    *coordPitch(flowField, pix) = flow;
}

} // namespace gpu
} // namespace flowfilter
