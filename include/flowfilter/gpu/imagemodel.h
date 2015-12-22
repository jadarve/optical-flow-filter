/**
 * \file imagemodel.h
 * \brief type declarations vision pipelines.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_IMAGEMODEL_H_
#define FLOWFILTER_GPU_IMAGEMODEL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/gpu/pipeline.h"

namespace flowfilter {
    namespace gpu {

        
        class ImageModel : public Stage {

        public:
            ImageModel();
            ImageModel(const int height, const int width);
            ~ImageModel();

        public:
            /**
             * \brief load uint8 input image to device
             */
            void loadImage(flowfilter::image_t& img);

            /**
             * \brief performs computation of brightness parameters
             */
            void compute();

            flowfilter::gpu::GPUImage getImageConstantDevice();
            flowfilter::gpu::GPUImage getImageGradientDevice();

        private:
            flowfilter::gpu::GPUImage __inputImage;
            flowfilter::gpu::GPUImage __imageConstant;
            flowfilter::gpu::GPUImage __imageGradient;
        };

    }; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_IMAGEMODEL_H_