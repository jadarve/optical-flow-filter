/**
 * \file image.h
 * \brief type declarations for image buffers
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_IMAGE_H_
#define FLOWFILTER_IMAGE_H_

#include <cstddef>

#include "flowfilter/osconfig.h"

// #pragma message ("MESSAGE FROM image.h: " XSTR(FLOWFILTER_API))

/**
 * \brief The namespace of this library.
 */
namespace flowfilter {

/**
 * \brief Image type
 *
 * This type encapsulates the relevant attributes of
 * an image buffer to enable its manipulation within
 * the library.
 *
 */
typedef struct {

    /** Image height in pixels */
    int height;

    /** Image width in pixels */
    int width;

    /** Image depth in pixels */
    int depth;

    /** Row pitch in bytes */
    std::size_t pitch;

    /** Item size in bytes */
    std::size_t itemSize;

    /** Memory buffer pointer */
    void* data;
} image_t;


/**
 * \brief Creates a new image. Allocates memory for storing pixel data.
 *
 * \param[in] height image height in pixels.
 * \param[in] width image width in pixels.
 * \param[in] pixelSize pixel size in bytes.
 */
FLOWFILTER_API image_t createImage(const int height, const int width, const size_t pixelSize);

/**
 * \brief Creates a new image. Allocates memory for storing pixel data.
 *
 * \param[in] height image height in pixels.
 * \param[in] width image width in pixels.
 * \param[in] depth image depth in pixels
 * \param[in] pixelSize pixel size in bytes.
 */
FLOWFILTER_API image_t createImage(const int height, const int width, const int depth, const size_t pixelSize);

/**
 * \brief Destroys an image. Deallocates image memory buffer.
 *
 * Releases the memory allocated to store the buffer
 * containing image data. image.data is set to std::nullptr
 * after deallocation.
 */
FLOWFILTER_API void destroyImage(image_t& image);


}; // namespace flowfilter

#endif