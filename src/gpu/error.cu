/**
 * \file error.cu
 * \brief error utility functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>

#include "flowfilter/gpu/error.h"

namespace flowfilter {
namespace gpu {

void assertError(cudaError_t errorCode, const char* file,
    int line, bool abort) {

    if(errorCode != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(errorCode) <<
            " at " << file << " : " << line << std::endl;

        if(abort) exit(errorCode);
    }
}

}; // namespace gpu
}; // namespace flowfilter