/**
 * \file image.h
 * \brief type declarations for GPU image buffers.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_IMAGE_H_
#define FLOWFILTER_GPU_IMAGE_H_

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/image.h"

namespace flowfilter {
    namespace gpu {

        /**
         * \brief struct to encapsulated image information.
         *
         * This structure is used internally in the kernel and
         * device functions.
         */
        template<typename T>
        struct gpuimage_t {

            /** height in pixels */
            int height;

            /** width in pixels */
            int width;

            /** row pitch in bytes*/
            std::size_t pitch;

            /** memory buffer*/
            T* data;
        };


        /*! \brief GPU Image container.
         */
        class GPUImage {

        public:
            GPUImage();
            GPUImage(const int height, const int width,
                const int depth = 1, const int itemSize = sizeof(char));

            ~GPUImage();

        public:
            int height() const;
            int width() const;
            int depth() const;
            int pitch() const;
            int itemSize() const;

            void* data();

            template<typename T>
            inline gpuimage_t<T> wrap() {
                gpuimage_t<T> img;
                img.height = __height;
                img.width = __width;
                img.pitch = __pitch;
                img.data = (T*)__ptr_dev.get();
                return img;
            }


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


        class GPUTexture {

        public:
            GPUTexture();
            GPUTexture( flowfilter::gpu::GPUImage& img, cudaChannelFormatKind format);
            GPUTexture( flowfilter::gpu::GPUImage& img,
                        cudaChannelFormatKind format,
                        cudaTextureReadMode readMode);
            GPUTexture( flowfilter::gpu::GPUImage& img,
                        cudaChannelFormatKind format,
                        cudaTextureAddressMode addressMode,
                        cudaTextureFilterMode filterMode,
                        cudaTextureReadMode readMode);

            ~GPUTexture();

        public:
            cudaTextureObject_t getTextureObject();
            flowfilter::gpu::GPUImage getImage();


        private:
            void configure( cudaChannelFormatKind format,
                            cudaTextureAddressMode addressMode,
                            cudaTextureFilterMode filterMode,
                            cudaTextureReadMode readMode);

        private:
            /** image buffer in GPU memory space */
            flowfilter::gpu::GPUImage __image;

            /** CUDA texture object */
            cudaTextureObject_t __texture;

            // tells if the created texture is valid
            bool __validTexture;

            /** reference counter for this object */
            std::shared_ptr<int> __refCounter;
        };

    }; // namespace gpu
}; // namespace flowfilter


#endif // FLOWFILTER_GPU_IMAGE_H_