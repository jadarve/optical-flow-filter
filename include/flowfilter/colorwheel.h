/**
 * \file colorwheel.h
 * \brief contains RGB values of a color wheel for color encoding optical flow fields
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_COLORWHEEL_H_
#define FLOWFILTER_COLORWHEEL_H_

#include "flowfilter/osconfig.h"
#include "flowfilter/image.h"

namespace flowfilter {

    /** \brief Contains the RBG values of a color wheel texture to encode optical flow fields
     *
     * The color wheel texture contains RGB values for each
     */
    FLOWFILTER_API extern unsigned char COLOR_WHEEL_D[];

    FLOWFILTER_API const int COLOR_WHEEL_HEIGHT = 256;
    FLOWFILTER_API const int COLOR_WHEEL_WIDTH = 256;
    FLOWFILTER_API const int COLOR_WHEEL_DEPTH = 4;


    FLOWFILTER_API image_t getColorWheelRGBA();

}; // namespace flowfilter

#endif // FLOWFILTER_COLORWHEEL_H_