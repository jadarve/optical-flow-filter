/**
 * \file gpu_deleter.h
 * \brief contains a memory deleter for GPU memory buffers.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_GPU_DELETER_H_
#define FLOWFILTER_GPU_GPU_DELETER_H_

#include <cuda.h>
#include <cuda_runtime.h>

namespace flowfilter {
namespace gpu {

template<typename T>
struct gpu_deleter {
    void operator()(T* p) {
        
        if(p) {
            // dont put std::cerr here, in Python appears an error when importing module
            // std::cerr << "gpu_deleter: calling cudaFree()" << std::endl;
            
            cudaError_t err = cudaFree(p);
            if(err != cudaSuccess) {
                std::cerr << "ERROR: gpu_deleter: "
                        << cudaGetErrorString(err) << std::endl;
            }
        }
    }
};

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_GPU_DELETER_H_
