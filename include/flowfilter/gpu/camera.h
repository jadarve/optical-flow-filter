/**
 * \file camera.h
 * \brief Camera model classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_CAMERA_H_
#define FLOWFILTER_GPU_CAMERA_H_

namespace flowfilter {
namespace gpu {

#include "flowfilter/osconfig.h"

/**
 * \brief Perspective camera intrinsic parameters.
 */
typedef struct {
    float alphaX;
    float alphaY;
    float centerX;
    float centerY;
} perspectiveCamera;


/**
 * \brief creates a perspective camera from sensor parameters
 *
 * \param focalLength focal length in millimiters
 * \param height image height in pixels
 * \param width image width in pixels
 * \param sensorHeight sensor height in millimiters
 * \param sensorWidth sensor width in millimiters
 */
FLOWFILTER_API perspectiveCamera createPerspectiveCamera(
    const float focalLength, const int height, const int width,
    const float sensorHeight, const float sensorWidth);

} // namespace gpu
} // namespace flowfilter

#endif /* FLOWFILTER_GPU_CAMERA_H_ */
