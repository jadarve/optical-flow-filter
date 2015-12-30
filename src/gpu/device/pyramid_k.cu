/**
 * \file pyramid_k.cu
 * \brief Kernel declaration for computing image and flow pyramids
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include "flowfilter/gpu/device/image_k.h"
#include "flowfilter/gpu/device/pyramid_k.h"

namespace flowfilter {
namespace gpu {

__global__ void imageDownX_uint8_k(cudaTextureObject_t inputImage,
    gpuimage_t<unsigned char> imageDown) {

    const int height = imageDown.height;
    const int width = imageDown.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
        blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // NOTE: the texture coordinates in X direction need to be multiplied
    //  by 2. This because image width is double with respect to
    //  the width of the output imageDown

    // image values around pixel in X direction (normalized [0, 1])
    const float img_m = tex2D<float>(inputImage, 2*pix.x -1, pix.y);
    const float img_0 = tex2D<float>(inputImage, 2*pix.x, pix.y);
    const float img_p = tex2D<float>(inputImage, 2*pix.x +1, pix.y);

    // smoothed image
    float imgSmoothed = 0.5*img_0 + 0.25*(img_m + img_p);

    //#################################
    // PACK RESULTS
    //#################################
    *coordPitch(imageDown, pix) = (unsigned char)(255*imgSmoothed);
    
}


__global__ void imageDownY_uint8_k(cudaTextureObject_t inputImage,
    gpuimage_t<unsigned char> imageDown) {

    const int height = imageDown.height;
    const int width = imageDown.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
        blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // NOTE: the texture coordinates in Y direction need to be multiplied
    //  by 2. This because image height is double with respect to
    //  the height of the output imageDown

    // image values around pixel in Y direction (normalized [0, 1])
    const float img_m = tex2D<float>(inputImage, pix.x, 2*pix.y -1);
    const float img_0 = tex2D<float>(inputImage, pix.x, 2*pix.y);
    const float img_p = tex2D<float>(inputImage, pix.x, 2*pix.y +1);

    // smoothed image
    float imgSmoothed = 0.5*img_0 + 0.25*(img_m + img_p);

    //#################################
    // PACK THE RESULTS
    //#################################
    *coordPitch(imageDown, pix) = (unsigned char)(255*imgSmoothed);
}


__global__ void imageDownX_float_k(cudaTextureObject_t inputImage,
    gpuimage_t<float> imageDown) {

    const int height = imageDown.height;
    const int width = imageDown.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
        blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // NOTE: the texture coordinates in X direction need to be multiplied
    //  by 2. This because image width is double with respect to
    //  the width of the output imageDown

    // image values around pixel in X direction
    const float img_m = tex2D<float>(inputImage, 2*pix.x -1, pix.y);
    const float img_0 = tex2D<float>(inputImage, 2*pix.x, pix.y);
    const float img_p = tex2D<float>(inputImage, 2*pix.x +1, pix.y);

    // smoothed image
    float imgSmoothed = 0.5*img_0 + 0.25*(img_m + img_p);

    //#################################
    // PACK RESULTS
    //#################################
    *coordPitch(imageDown, pix) = imgSmoothed;
    
}


__global__ void imageDownY_float_k(cudaTextureObject_t inputImage,
    gpuimage_t<float> imageDown) {

    const int height = imageDown.height;
    const int width = imageDown.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
        blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    // NOTE: the texture coordinates in Y direction need to be multiplied
    //  by 2. This because image height is double with respect to
    //  the height of the output imageDown

    // image values around pixel in Y direction
    const float img_m = tex2D<float>(inputImage, pix.x, 2*pix.y -1);
    const float img_0 = tex2D<float>(inputImage, pix.x, 2*pix.y);
    const float img_p = tex2D<float>(inputImage, pix.x, 2*pix.y +1);

    // smoothed image
    float imgSmoothed = 0.5*img_0 + 0.25*(img_m + img_p);

    //#################################
    // PACK THE RESULTS
    //#################################
    *coordPitch(imageDown, pix) = imgSmoothed;
}

}; // namespace gpu
}; // namespace flowfilter
