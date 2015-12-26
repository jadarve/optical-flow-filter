/**
 * \file propagation_k.cu
 * \brief Kernel declarations for flow propagation methods.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include "flowfilter/gpu/device/image_k.h"
#include "flowfilter/gpu/device/propagation_k.h"


namespace flowfilter {
namespace gpu {


__global__ void flowPropagateX_k(cudaTextureObject_t inputFlow,
                                 gpuimage_t<float2> flowPropagated,
                                 const float dt, const int border) {
    
    const int height = flowPropagated.height;
    const int width = flowPropagated.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // flow values around pixel in X direction
    const float2 flow_m = tex2D<float2>(inputFlow, pix.x -1, pix.y);
    const float2 flow_0 = tex2D<float2>(inputFlow, pix.x, pix.y);
    const float2 flow_p = tex2D<float2>(inputFlow, pix.x +1, pix.y);

    // central difference of U_abs
    float Uabs_central = abs(flow_p.x) - abs(flow_m.x);

    // dominant velocity
    float Ud = Uabs_central > 0.0f? flow_p.x : flow_m.x;

    // forward and backward differences of U in X
    float ux_p = flow_p.x - flow_0.x;
    float ux_m = flow_0.x - flow_m.x;

    // forward and backward differences of V in X
    float vx_p = flow_p.y - flow_0.y;
    float vx_m = flow_0.y - flow_m.y;

    // propagation in X
    float2 flowPropU = flow_0;
    flowPropU.x -= dt*Ud* (Ud >= 0.0f? ux_m : ux_p);
    flowPropU.y -= dt*Ud* (Ud >= 0.0f? vx_m : vx_p);

    //#################################
    // PACK RESULTS
    //#################################
    *coordPitch(flowPropagated, pix) = flowPropU;
}


__global__ void flowPropagateY_k(cudaTextureObject_t inputFlow,
                                 gpuimage_t<float2> flowPropagated,
                                 const float dt, const int border) {

    const int height = flowPropagated.height;
    const int width = flowPropagated.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }


    // flow values around pixel in Y direction
    const float2 flow_m = tex2D<float2>(inputFlow, pix.x, pix.y -1);
    const float2 flow_0 = tex2D<float2>(inputFlow, pix.x, pix.y);
    const float2 flow_p = tex2D<float2>(inputFlow, pix.x, pix.y +1);

    // central difference of V_abs
    float Vabs_central = abs(flow_p.y) - abs(flow_m.y);

    // dominant velocity
    float Vd = Vabs_central > 0.0f? flow_p.y : flow_m.y;

    // forward and backward differences of U in Y
    float uy_p = flow_p.x - flow_0.x;
    float uy_m = flow_0.x - flow_m.x;

    // forward and backward differences of V in Y
    float vy_p = flow_p.y - flow_0.y;
    float vy_m = flow_0.y - flow_m.y;

    // propagation in Y
    float2 flowPropV = flow_0;
    flowPropV.x -= dt*Vd* (Vd >= 0.0f? uy_m : uy_p);
    flowPropV.y -= dt*Vd* (Vd >= 0.0f? vy_m : vy_p);

    //#################################
    // BORDER REMOVAL
    //#################################
    const unsigned int inRange = (pix.x >= border && pix.x < width - border) &&
                                 (pix.y >= border && pix.y < height - border);

    // if the pixel coordinate lies on the image border,
    // take the original value of flow (flow_0) as the propagated flow
    flowPropV.x = inRange? flowPropV.x : flow_0.x;
    flowPropV.y = inRange? flowPropV.y : flow_0.y;

    //#################################
    // PACK THE RESULTS
    //#################################
    *coordPitch(flowPropagated, pix) = flowPropV;
}

}; // namespace gpu
}; // namespace flowfilter