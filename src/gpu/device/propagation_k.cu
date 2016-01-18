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
    // BORDER REMOVAL
    //#################################
    const unsigned int inRange = (pix.x >= border && pix.x < width - border) &&
                                 (pix.y >= border && pix.y < height - border);

    // if the pixel coordinate lies on the image border,
    // take the original value of flow (flow_0) as the propagated flow
    flowPropU.x = inRange? flowPropU.x : flow_0.x;
    flowPropU.y = inRange? flowPropU.y : flow_0.y;


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


__global__ void flowPropagatePayloadX_k(cudaTextureObject_t inputFlow,
                                        gpuimage_t<float2> flowPropagated,
                                        cudaTextureObject_t scalarPayload,
                                        gpuimage_t<float> scalarPropagated,
                                        cudaTextureObject_t vectorPayload,
                                        gpuimage_t<float2> vectorPropagated,
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
    const float Uabs_central = abs(flow_p.x) - abs(flow_m.x);

    // dominant velocity
    const float Ud = Uabs_central > 0.0f? flow_p.x : flow_m.x;

    // forward and backward differences of U in X
    const float ux_p = flow_p.x - flow_0.x;
    const float ux_m = flow_0.x - flow_m.x;

    // forward and backward differences of V in X
    const float vx_p = flow_p.y - flow_0.y;
    const float vx_m = flow_0.y - flow_m.y;

    // propagation in X
    float2 flowPropU = flow_0;
    flowPropU.x -= dt*Ud* (Ud >= 0.0f? ux_m : ux_p);
    flowPropU.y -= dt*Ud* (Ud >= 0.0f? vx_m : vx_p);


    //#################################
    // SCALAR PAYLOAD PROPAGATION
    //#################################
    const float load1_m = tex2D<float>(scalarPayload, pix.x -1, pix.y);
    const float load1_0 = tex2D<float>(scalarPayload, pix.x, pix.y);
    const float load1_p = tex2D<float>(scalarPayload, pix.x +1, pix.y);

    // forward and backward differences
    const float lx1_p = load1_p - load1_0;
    const float lx1_m = load1_0 - load1_m;

    float loadProp1 = load1_0;
    loadProp1 -= dt*Ud* (Ud >= 0.0f? lx1_m : lx1_p);


    //#################################
    // VECTOR PAYLOAD PROPAGATION
    //#################################
    const float2 load2_m = tex2D<float2>(vectorPayload, pix.x -1, pix.y);
    const float2 load2_0 = tex2D<float2>(vectorPayload, pix.x, pix.y);
    const float2 load2_p = tex2D<float2>(vectorPayload, pix.x +1, pix.y);

    // forward and backward differences
    const float2 lx2_p = make_float2(load2_p.x - load2_0.x, load2_p.y - load2_0.y);
    const float2 lx2_m = make_float2(load2_0.x - load2_m.x, load2_0.y - load2_m.y);

    float2 loadProp2 = load2_0;
    loadProp2.x -= dt*Ud* (Ud >= 0.0f? lx2_m.x : lx2_p.x);
    loadProp2.y -= dt*Ud* (Ud >= 0.0f? lx2_m.y : lx2_p.y);


    //#################################
    // BORDER REMOVAL
    //#################################
    const unsigned int inRange = (pix.x >= border && pix.x < width - border) &&
                                 (pix.y >= border && pix.y < height - border);

    // if the pixel coordinate lies on the image border,
    // take the original value of flow (flow_0) as the propagated flow
    flowPropU.x = inRange? flowPropU.x : flow_0.x;
    flowPropU.y = inRange? flowPropU.y : flow_0.y;

    loadProp1 = inRange? loadProp1 : load1_0;

    loadProp2.x = inRange? loadProp2.x : load2_0.x;
    loadProp2.y = inRange? loadProp2.y : load2_0.y;

    //#################################
    // PACK RESULTS
    //#################################
    *coordPitch(flowPropagated, pix) = flowPropU;
    *coordPitch(scalarPropagated, pix) = loadProp1;
    *coordPitch(vectorPropagated, pix) = loadProp2;
}


__global__ void flowPropagatePayloadY_k(cudaTextureObject_t inputFlow,
                                        gpuimage_t<float2> flowPropagated,
                                        cudaTextureObject_t scalarPayload,
                                        gpuimage_t<float> scalarPropagated,
                                        cudaTextureObject_t vectorPayload,
                                        gpuimage_t<float2> vectorPropagated,
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
    const float Vabs_central = abs(flow_p.y) - abs(flow_m.y);

    // dominant velocity
    const float Vd = Vabs_central > 0.0f? flow_p.y : flow_m.y;

    // forward and backward differences of U in Y
    const float uy_p = flow_p.x - flow_0.x;
    const float uy_m = flow_0.x - flow_m.x;

    // forward and backward differences of V in Y
    const float vy_p = flow_p.y - flow_0.y;
    const float vy_m = flow_0.y - flow_m.y;

    // propagation in Y
    float2 flowPropV = flow_0;
    flowPropV.x -= dt*Vd* (Vd >= 0.0f? uy_m : uy_p);
    flowPropV.y -= dt*Vd* (Vd >= 0.0f? vy_m : vy_p);


    //#################################
    // FLOAT1 PAYLOAD PROPAGATION
    //#################################
    const float load1_m = tex2D<float>(scalarPayload, pix.x, pix.y -1);
    const float load1_0 = tex2D<float>(scalarPayload, pix.x, pix.y);
    const float load1_p = tex2D<float>(scalarPayload, pix.x, pix.y +1);

    // forward and backward differences
    const float ly1_p = load1_p - load1_0;
    const float ly1_m = load1_0 - load1_m;

    float loadProp1 = load1_0;
    loadProp1 -= dt*Vd* (Vd >= 0.0f? ly1_m : ly1_p);


    //#################################
    // FLOAT2 PAYLOAD PROPAGATION
    //#################################
    const float2 load2_m = tex2D<float2>(vectorPayload, pix.x, pix.y -1);
    const float2 load2_0 = tex2D<float2>(vectorPayload, pix.x, pix.y);
    const float2 load2_p = tex2D<float2>(vectorPayload, pix.x, pix.y +1);

    // forward and backward differences
    const float2 ly2_p = make_float2(load2_p.x - load2_0.x, load2_p.y - load2_0.y);
    const float2 ly2_m = make_float2(load2_0.x - load2_m.x, load2_0.y - load2_m.y);

    float2 loadProp2 = load2_0;
    loadProp2.x -= dt*Vd* (Vd >= 0.0f? ly2_m.x : ly2_p.x);
    loadProp2.y -= dt*Vd* (Vd >= 0.0f? ly2_m.y : ly2_p.y);


    //#################################
    // BORDER REMOVAL
    //#################################
    const unsigned int inRange = (pix.x >= border && pix.x < width - border) &&
                                 (pix.y >= border && pix.y < height - border);

    // if the pixel coordinate lies on the image border,
    // take the original value of flow (flow_0) as the propagated flow
    flowPropV.x = inRange? flowPropV.x : flow_0.x;
    flowPropV.y = inRange? flowPropV.y : flow_0.y;

    loadProp1 = inRange? loadProp1 : load1_0;

    loadProp2.x = inRange? loadProp2.x : load2_0.x;
    loadProp2.y = inRange? loadProp2.y : load2_0.y;


    //#################################
    // PACK RESULTS
    //#################################
    *coordPitch(flowPropagated, pix) = flowPropV;
    *coordPitch(scalarPropagated, pix) = loadProp1;
    *coordPitch(vectorPropagated, pix) = loadProp2;
}

}; // namespace gpu
}; // namespace flowfilter