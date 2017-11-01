/**
 * \file imagemodel_k.cu
 * \brief Kernel declarations for image model computation.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include "flowfilter/gpu/device/image_k.h"
#include "flowfilter/gpu/device/imagemodel_k.h"


namespace flowfilter {
namespace gpu {

//######################
// 5 support
//######################
#define IMS_R 2
#define IMS_W 5

__constant__ float smooth_mask[] = {0.0625,  0.25,    0.375,   0.25,    0.0625};
__constant__ float diff_mask[] = {-0.125, -0.25, 0, 0.25, 0.125};


/**
 * \brief Apply a smooth mask to input image in X and Y directions.
 *
 * NOTE:    reading float, either from a float image or a normalized
 *          image is faster than reading unsigned char directly.
 */
__global__ void imagePrefilter_k(cudaTextureObject_t inputImage,
        gpuimage_t<float2> imgPrefiltered) {

    const int height = imgPrefiltered.height;
    const int width = imgPrefiltered.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    //#################################
    // SMOOTHING IN X
    //#################################
    float smooth_x = 0.0f;

    #pragma unroll
    for(int c = -IMS_R; c <= IMS_R; c ++) {
        smooth_x += smooth_mask[c + IMS_R] * tex2D<float>(inputImage, pix.x + c, pix.y);
    }

    //#################################
    // SMOOTHING IN Y
    //#################################
    float smooth_y = 0.0f;

    #pragma unroll
    for(int r = -IMS_R; r <= IMS_R; r ++) {
        smooth_y += smooth_mask[r + IMS_R] * tex2D<float>(inputImage, pix.x, pix.y + r);
    }

    //#################################
    // PACK RESULTS
    //#################################
    // {smooth_y, smooth_x}
    *coordPitch(imgPrefiltered, pix) = make_float2(smooth_y, smooth_x);
}


/**
 * \brief Compute image gradient and constant term from XY smoothed image.
 */
__global__ void imageModel_k(cudaTextureObject_t imgPrefiltered,
        gpuimage_t<float> imgConstant,
        gpuimage_t<float2> imgGradient) {


    const int height = imgConstant.height;
    const int width = imgConstant.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // imgPrefiltered texture element
    float2 imElement;

    float diff_x = 0.0;
    float diff_y = 0.0;
    float smooth = 0.0;

    //#################################
    // DIFFERENCING IN X
    //#################################

    #pragma unroll
    for(int c = -IMS_R; c <= IMS_R; c ++) {
        // texture coordinate
        imElement = tex2D<float2>(imgPrefiltered, pix.x + c, pix.y);

        // convolution with difference kernel
        diff_x += diff_mask[c + IMS_R]*imElement.x;

        // convolution with smooth kernel
        smooth += smooth_mask[c + IMS_R]*imElement.x;
    }

    //#################################
    // DIFFERENCING IN Y
    //#################################

    #pragma unroll
    for(int r = -IMS_R; r <= IMS_R; r ++) {
        imElement = tex2D<float2>(imgPrefiltered, pix.x, pix.y + r);

        // convolution difference kernel
        diff_y += diff_mask[r + IMS_R]*imElement.y;
    }

    //#################################
    // PACK RESULTS
    //#################################
    // {diff_x, diff_y}
    *coordPitch(imgGradient, pix) = make_float2(diff_x, diff_y);
    *coordPitch(imgConstant, pix) = smooth;
}

}; // namespace gpu
}; // namespace flowfilter