/**
 * \file misc_k.cu
 * \brief Miscellaneous kernel declarations.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include "flowfilter/gpu/device/image_k.h"
#include "flowfilter/gpu/device/misc_k.h"


namespace flowfilter {
namespace gpu {

/**
 * \brief Multiplies an input vector field by a scalar constant.
 */
__global__ void scalarProductF2_k(gpuimage_t<float2> inputField,
                                  const float scalar,
                                  gpuimage_t<float2> outputField) {

    const int height = inputField.height;
    const int width = inputField.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // read input vector
    float2 v = *coordPitch(inputField, pix);


    // write (scalar * v) in outputField
    *coordPitch(outputField, pix) = make_float2(scalar * v.x, scalar * v.y);
}

}; // namespace gpu
}; // namespace flowfilter
