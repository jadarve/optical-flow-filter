/**
 * \file imagemodel_k.h
 * \brief Kernel declarations for image model computation.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_IMAGEMODEL_K_H_
#define FLOWFILTER_GPU_IMAGEMODEL_K_H_


#include "flowfilter/gpu/image.h"


namespace flowfilter {
namespace gpu {

/**
 * \brief Apply a smooth mask to input image in X and Y directions.
 */
__global__ void imagePrefilter_k(cudaTextureObject_t inputImage,
                                 gpuimage_t<float2> imgPrefiltered);

/**
 * \brief Compute image gradient and constant term from XY smoothed image.
 */
__global__ void imageModel_k(cudaTextureObject_t imgPrefiltered,
                             gpuimage_t<float> imgConstant,
                             gpuimage_t<float2> imgGradient);

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_IMAGEMODEL_K_H_