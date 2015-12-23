/**
 * \file image_k.h
 * \brief Device functions for image manipulation
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_IMAGE_K_H_
#define FLOWFILTER_GPU_IMAGE_K_H_

#include <memory>

#include "flowfilter/gpu/image.h"

namespace flowfilter {
    namespace gpu {

        template<typename T>
        __device__ __forceinline__ T* coordPitch(gpuimage_t<T> img, const int2 pix) {
            return (T*)((char*)img.data + pix.y*img.pitch) + pix.x;
        }

    }; // namespace gpu
}; // namespace flowfilter


#endif // FLOWFILTER_GPU_IMAGE_K_H_