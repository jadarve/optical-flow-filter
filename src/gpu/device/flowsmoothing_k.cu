/**
 * \file flowsmoothing_k.h
 * \brief Kernel declarations for image model computation.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include "flowfilter/gpu/device/image_k.h"
#include "flowfilter/gpu/device/flowsmoothing_k.h"


namespace flowfilter {
namespace gpu {

//######################
// 5 support
//######################
#define FSS_R 2
__constant__ float flowSmooth5_k[] = {0.2, 0.2, 0.2, 0.2, 0.2};

__global__ void flowSmoothX_k(cudaTextureObject_t inputFlow,
        gpuimage_t<float2> flowSmooth) {

    const int height = flowSmooth.height;
    const int width = flowSmooth.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    //#################################
    // SMOOTHING IN X
    //#################################
    float2 smooth_x = make_float2(0.0f, 0.0f);
    
    #pragma unroll
    for(int c = -FSS_R; c <= FSS_R; c ++) {
        float2 flow = tex2D<float2>(inputFlow, pix.x + c, pix.y);
        float coeff = flowSmooth5_k[c + FSS_R];

        smooth_x.x += coeff*flow.x;
        smooth_x.y += coeff*flow.y;
    }

    //#################################
    // PACK RESULTS
    //#################################
    *coordPitch(flowSmooth, pix) = smooth_x;
}

__global__ void flowSmoothY_k(cudaTextureObject_t inputFlow,
        gpuimage_t<float2> flowSmooth) {

    const int height = flowSmooth.height;
    const int width = flowSmooth.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    //#################################
    // SMOOTHING IN Y
    //#################################
    float2 smooth_y = make_float2(0.0f, 0.0f);

    #pragma unroll
    for(int r = -FSS_R; r <= FSS_R; r ++) {
        float2 flow = tex2D<float2>(inputFlow, pix.x, pix.y  + r);
        float coeff = flowSmooth5_k[r + FSS_R];

        smooth_y.x += coeff*flow.x;
        smooth_y.y += coeff*flow.y;
    }

    //#################################
    // PACK RESULTS
    //#################################
    *coordPitch(flowSmooth, pix) = smooth_y;
}

}; // namespace gpu
}; // namespace flowfilter