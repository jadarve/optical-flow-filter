/**
 * \file util.h
 * \brief Miscelaneous utility functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_UTIL_H_
#define FLOWFILTER_GPU_UTIL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/osconfig.h"

namespace flowfilter {
namespace gpu {

/**
 * \brief Configure kernel grid size according to image and block size.
 */
FLOWFILTER_API void configureKernelGrid(const int height, const int width,
    const dim3 block, dim3& grid);


}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_UTIL_H_