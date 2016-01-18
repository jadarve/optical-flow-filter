/**
 * \file update_k.cu
 * \brief Kernel declarations for optical flow update computation.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include "flowfilter/gpu/device/image_k.h"
#include "flowfilter/gpu/device/update_k.h"


namespace flowfilter {
namespace gpu {

__global__ void flowUpdate_k(gpuimage_t<float> newImage, 
    gpuimage_t<float2> newImageGradient,
    gpuimage_t<float> oldImage, gpuimage_t<float2> oldFlow,
    gpuimage_t<float> imageUpdated, gpuimage_t<float2> flowUpdated,
    const float gamma, const float maxflow) {


    const int height = flowUpdated.height;
    const int width = flowUpdated.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // read elements from the different arrays
    float2 a1 = *coordPitch(newImageGradient, pix);
    float a0 = *coordPitch(newImage, pix);
    float a0old = *coordPitch(oldImage, pix);
    float2 ofOld = *coordPitch(oldFlow, pix);

    //#################################
    // FLOW UPDATE
    //#################################
    // temporal derivative
    float Yt = a0old - a0;

    float ax2 = a1.x*a1.x;
    float ay2 = a1.y*a1.y;

    // elements of the adjucate matrix of M
    float N00 = gamma + ay2;
    float N01 = -a1.x*a1.y;
    float N10 = N01;
    float N11 = gamma + ax2;

    // reciprocal determinant of M
    float rdetM = 1.0f / (gamma*(gamma + ax2 + ay2));

    // q vector components
    float qx = gamma*ofOld.x + a1.x*Yt;
    float qy = gamma*ofOld.y + a1.y*Yt;

    // computes the updated optical flow
    float2 ofNew = make_float2( (N00*qx + N01*qy)*rdetM,
                                (N10*qx + N11*qy)*rdetM);

    // truncates the flow to lie in its allowed interval
    ofNew.x = max(-maxflow, min(ofNew.x, maxflow));
    ofNew.y = max(-maxflow, min(ofNew.y, maxflow));

    // sanitize the output
    ofNew.x = isinf(ofNew.x) + isnan(ofNew.x) > 0? 0.0f : ofNew.x;
    ofNew.y = isinf(ofNew.y) + isnan(ofNew.y) > 0? 0.0f : ofNew.y;


    //#################################
    // PACK RESULTS
    //#################################
    *coordPitch(flowUpdated, pix) = ofNew;
    *coordPitch(imageUpdated, pix) = a0;
}


__global__ void deltaFlowUpdate_k(gpuimage_t<float> newImage,
                                  gpuimage_t<float2> newImageGradient,
                                  gpuimage_t<float> oldImage,
                                  gpuimage_t<float2> oldDeltaFlow,
                                  cudaTextureObject_t oldFlowTexture,
                                  gpuimage_t<float> imageUpdated,
                                  gpuimage_t<float2> deltaFlowUpdated,
                                  gpuimage_t<float2> flowUpdated,
                                  const float gamma, const float maxflow) {


    const int height = flowUpdated.height;
    const int width = flowUpdated.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // read elements from the different arrays
    float2 a1 = *coordPitch(newImageGradient, pix);
    float a0 = *coordPitch(newImage, pix);
    float a0old = *coordPitch(oldImage, pix);
    float2 deltaFlowOld = *coordPitch(oldDeltaFlow, pix);

    //#################################
    // FLOW UPDATE
    //#################################
    // temporal derivative
    float Yt = a0old - a0;

    float ax2 = a1.x*a1.x;
    float ay2 = a1.y*a1.y;

    // elements of the adjucate matrix of M
    float N00 = gamma + ay2;
    float N01 = -a1.x*a1.y;
    float N10 = N01;
    float N11 = gamma + ax2;

    // reciprocal determinant of M
    float rdetM = 1.0f / (gamma*(gamma + ax2 + ay2));

    // q vector components
    float qx = gamma*deltaFlowOld.x + a1.x*Yt;
    float qy = gamma*deltaFlowOld.y + a1.y*Yt;

    // computes updated optical flow
    float2 dFlowNew = make_float2( (N00*qx + N01*qy)*rdetM,
                                (N10*qx + N11*qy)*rdetM);

    // sanitize output
    dFlowNew.x = isinf(dFlowNew.x) + isnan(dFlowNew.x) > 0? 0.0f : dFlowNew.x;
    dFlowNew.y = isinf(dFlowNew.y) + isnan(dFlowNew.y) > 0? 0.0f : dFlowNew.y;

    // truncates dflow to lie in its allowed interval
    dFlowNew.x = max(-0.5f*maxflow, min(dFlowNew.x, 0.5f*maxflow));
    dFlowNew.y = max(-0.5f*maxflow, min(dFlowNew.y, 0.5f*maxflow));


    //#################################
    // OPTICAL FLOW COMPUTATION
    //#################################
    
    // read upper level flow
    // normalized texture coordinates
    float u = (float)pix.x / (float)(width -1);
    float v = (float)pix.y / (float)(height -1);

    // linear interpolation of flow value
    float2 fup = tex2D<float2>(oldFlowTexture, u, v);
    float2 flowUp = make_float2(2.0*fup.x, 2.0*fup.y);

    // update upsampled flow from top level
    float2 flowNew = make_float2(dFlowNew.x + flowUp.x,
        dFlowNew.y + flowUp.y);

    // truncates flow to lie in its allowed interval
    flowNew.x = max(-maxflow, min(flowNew.x, maxflow));
    flowNew.y = max(-maxflow, min(flowNew.y, maxflow));


    //#################################
    // PACK RESULTS
    //#################################
    *coordPitch(deltaFlowUpdated, pix) = dFlowNew;
    *coordPitch(flowUpdated, pix) = flowNew;
    *coordPitch(imageUpdated, pix) = a0;
}

}; // namespace gpu
}; // namespace flowfilter

