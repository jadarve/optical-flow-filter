/**
 * \file error.h
 * \brief error utility functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_ERROR_H_
#define FLOWFILTER_GPU_ERROR_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/osconfig.h"

// #pragma message ("MESSAGE FROM error.h: " XSTR(FLOWFILTER_API))

namespace flowfilter {
namespace gpu {


/**
 * \biref check the execution of a cuda call is successful.
 */
#define checkError(ans) { assertError(ans, __FILE__, __LINE__);}

/**
 * \brief assert if errorCode is successful.
 */
FLOWFILTER_API void assertError(cudaError_t errorCode, const char* file,
    int line, bool abort = true);

        
}; // namespace gpu
}; // namespace flowfilter

#endif  // FLOWFILTER_GPU_ERROR_H_