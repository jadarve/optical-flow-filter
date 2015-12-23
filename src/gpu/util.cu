/**
 * \file util.cu
 * \brief Miscelaneous utility functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#include "flowfilter/gpu/util.h"

namespace flowfilter {
    namespace gpu {

        /**
         * \brief Configure kernel grid size according to image and block size.
         */
        void configureKernelGrid(const int height, const int width,
            const dim3 block, dim3& grid) {

            float w = width;
            float h = height;
            float x = block.x;
            float y = block.y;

            grid.x = (int)ceilf(w / x);
            grid.y = (int)ceilf(h / y);
            grid.z = 1;
        }

    }; // namespace gpu
}; // namespace flowfilter
