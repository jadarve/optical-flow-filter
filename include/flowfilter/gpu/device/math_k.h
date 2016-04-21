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


//#########################################################
// FLOAT-4
//#########################################################

/** float4 c = a + b; */
inline __device__ float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

/** a += b */
inline __device__ void operator +=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

/** float4 c = a - b; */
inline __device__ float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

/** float4 a; float b; float4 c = a * b */
inline __device__ float4 operator*(const float4& a, const float b) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

/** float4 a; float b; float4 c = b * a */
inline __device__ float4 operator*(const float b, const float4& a) {
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

/** float4 a; float4 c; c -= a; */
inline __device__ void operator-=(float4& a, const float4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

//#####################################
// FUNCTIONS
//#####################################

} // namespace gpu
} // namespace flowfilter

#endif /* FLOWFILTER_GPU_MATH_K_H_ */