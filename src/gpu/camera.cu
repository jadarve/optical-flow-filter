/**
 * \file camera.h
 * \brief Camera model classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

 #include "flowfilter/gpu/camera.h"


namespace flowfilter {
namespace gpu {

perspectiveCamera createPerspectiveCamera(
    const float focalLength, const int height, const int width,
    const float sensorHeight, const float sensorWidth) {

    // range of allowed pixel coordinates
    int imgX = width -1;
    int imgY = height -1;

    perspectiveCamera cam;
    cam.alphaX = focalLength * imgX / sensorWidth;
    cam.alphaY = focalLength * imgY / sensorHeight;
    cam.centerX = 0.5f * imgX;
    cam.centerY = 0.5f * imgY;
    
    return cam;
}

}// namespace gpu
}// namespace flowfilter