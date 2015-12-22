/**
 * \file image.h
 * \brief type declarations for GPU image buffers.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_IMAGE_H_
#define FLOWFILTER_GPU_IMAGE_H_

#include <memory>

#include "flowfilter/image.h"
// #include "flowfilter/gpu/pipeline.h"

namespace flowfilter {
    namespace gpu {

        /*! \brief GPU Image container.
         */
        class GPUImage {

        public:
            GPUImage();
            GPUImage(const int height, const int width,
                const int depth = 1, const int itemSize = 4);

            ~GPUImage();

        public:
            int height() const;
            int width() const;
            int depth() const;
            int pitch() const;
            int itemSize() const;


            // TODO:: add stream parameter to support asynchrous copy
            /**
             * \brief upload an image in CPU to GPU memory space
             */
            void upload(flowfilter::image_t& img);

            /**
             * \brief download an image from GPU to CPU memory space
             */
            void download(flowfilter::image_t& img) const;


        private:
            std::size_t __width;
            std::size_t __height;
            std::size_t __depth;        // number of channels
            std::size_t __pitch;        // row pitch in bytes
            std::size_t __itemSize;     // item size in bytes
            std::shared_ptr<void> __ptr_dev;

        private:
            void allocate();
            bool compareShape(const flowfilter::image_t& img) const;

        };

    }; // namespace gpu
}; // namespace flowfilter


#endif // FLOWFILTER_GPU_IMAGE_H_