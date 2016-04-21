/**
 * \file propagation_k.h
 * \brief Kernel declarations for flow propagation methods.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_PROPAGATION_K_H_
#define FLOWFILTER_GPU_PROPAGATION_K_H_


#include "flowfilter/gpu/image.h"

#include "flowfilter/gpu/device/image_k.h"
#include "flowfilter/gpu/device/math_k.h"


namespace flowfilter {
namespace gpu {

__global__ void flowPropagateX_k(cudaTextureObject_t inputFlow,
                                 gpuimage_t<float2> flowPropagated,
                                 const float dt, const int border);


__global__ void flowPropagateY_k(cudaTextureObject_t inputFlow,
                                 gpuimage_t<float2> flowPropagated,
                                 const float dt, const int border);


__global__ void flowPropagatePayloadX_k(cudaTextureObject_t inputFlow,
                                        gpuimage_t<float2> flowPropagated,
                                        cudaTextureObject_t scalarPayload,
                                        gpuimage_t<float> scalarPropagated,
                                        cudaTextureObject_t vectorPayload,
                                        gpuimage_t<float2> vectorPropagated,
                                        const float dt, const int border);


__global__ void flowPropagatePayloadY_k(cudaTextureObject_t inputFlow,
                                        gpuimage_t<float2> flowPropagated,
                                        cudaTextureObject_t scalarPayload,
                                        gpuimage_t<float> scalarPropagated,
                                        cudaTextureObject_t vectorPayload,
                                        gpuimage_t<float2> vectorPropagated,
                                        const float dt, const int border);


template<typename T>
__global__ void LaxWendroffX_k(cudaTextureObject_t inputFlow,
                               cudaTextureObject_t inputImage,
                               gpuimage_t<T> propagatedImage,
                               const float dt) {

        const int height = propagatedImage.height;
        const int width = propagatedImage.width;

        // pixel coordinate
        const int2 pix = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y);

        if (pix.x >= width || pix.y >= height) {
                return;
        }

        const float2 flow = tex2D<float2>(inputFlow, pix.x, pix.y);

        // image values
        T img_m = tex2D<T>(inputImage, pix.x - 1, pix.y);
        T img_0 = tex2D<T>(inputImage, pix.x, pix.y);
        T img_p = tex2D<T>(inputImage, pix.x + 1, pix.y);


        // central difference
        T diff_0 = img_p - img_m;

        // second difference
        T diff2 = img_m - 2*img_0 + img_p;

        float R = dt*flow.x;

        // Lax-Wendroff scheme
        T img_prop = img_0 - 0.5*R*diff_0 + 0.5*R*R*diff2;

        //#################################
        // PACK RESULTS
        //#################################
        *coordPitch(propagatedImage, pix) = img_prop;
}


template<typename T>
__global__ void LaxWendroffY_k(cudaTextureObject_t inputFlow,
                               cudaTextureObject_t inputImage,
                               gpuimage_t<T> propagatedImage,
                               const float dt) {

        const int height = propagatedImage.height;
        const int width = propagatedImage.width;

        // pixel coordinate
        const int2 pix = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y);

        if (pix.x >= width || pix.y >= height) {
                return;
        }

        const float2 flow = tex2D<float2>(inputFlow, pix.x, pix.y);

        // image values
        T img_m = tex2D<T>(inputImage, pix.x, pix.y - 1);
        T img_0 = tex2D<T>(inputImage, pix.x, pix.y);
        T img_p = tex2D<T>(inputImage, pix.x, pix.y + 1);


        // central difference
        T diff_0 = img_p - img_m;

        // second difference
        T diff2 = img_m - 2.0*img_0 + img_p;

        float R = dt*flow.y;

        // Lax-Wendroff scheme
        T img_prop = img_0 - 0.5*R*diff_0 + 0.5*R*R*diff2;

        //#################################
        // PACK RESULTS
        //#################################
        *coordPitch(propagatedImage, pix) = img_prop;
}


}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_PROPAGATION_K_H_
