/**
 * \file display_k.cu
 * \brief Kernel declarations to convert optical flow fields to color representation.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include "flowfilter/gpu/device/image_k.h"

namespace flowfilter {
namespace gpu {


__global__ void flowToColor_k(gpuimage_t<float2> inputFlow,
                              cudaTextureObject_t colorWheel,
                              const float maxflow,
                              gpuimage_t<uchar4> colorFlow) {

    const int height = inputFlow.height;
    const int width = inputFlow.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // read optical flow
    const float2 flow = *coordPitch(inputFlow, pix);

    // normalized flow coordinates in range [0,1]
    float2 flowNormalized = make_float2((flow.x + maxflow)/(2*maxflow),
            (flow.y + maxflow)/(2*maxflow));


    // read color from color wheel texture
    uchar4 color = tex2D<uchar4>(colorWheel, flowNormalized.x, flowNormalized.y);

    // write color to output color buffer
    *coordPitch(colorFlow, pix) = color;
}


}; // namepsace gpu
}; // namespace flowfilter
