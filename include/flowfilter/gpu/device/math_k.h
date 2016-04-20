/**
 * \file camera_k.h
 * \brief Kernel math functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_MATH_K_H_
#define FLOWFILTER_GPU_MATH_K_H_


namespace flowfilter {
namespace gpu {


//#########################################################
// FLOAT-3
//#########################################################

//#####################################
// FUNCTIONS
//#####################################

inline __device__ float3 cross(const float3& a, const float3& b) {

    return make_float3( a.y*b.z - a.z*b.y,
                        a.z*b.x - a.x*b.z,
                        a.x*b.y - a.y*b.x);
}


} // namespace gpu
} // namespace flowfilter

#endif /* FLOWFILTER_GPU_MATH_K_H_ */