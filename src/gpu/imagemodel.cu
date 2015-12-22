/**
 * \file imagemodel.cu
 * \brief type declarations vision pipelines.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <exception>
#include <iostream>

#include "flowfilter/gpu/imagemodel.h"

namespace flowfilter {
    namespace gpu {

        //#################################################
        // ImageModel
        //#################################################
        ImageModel::ImageModel() {
            __configured = false;
        }

        /**
         * \brief creates an image model stage with a given input image
         *
         * This constructor internally calles configure() so that the
         * stage is ready to perform computations.
         */
        ImageModel::ImageModel(flowfilter::gpu::GPUImage inputImage) {
            
            __configured = false;
            setInputImage(inputImage);
            configure();
        }

        ImageModel::~ImageModel() {

            // nothing to do...
        }

        void ImageModel::configure() {

            int height = __inputImage.height();
            int width = __inputImage.width();

            // wraps __inputImage with unsigned char texture
            __inputImageTexture = GPUTexture(__inputImage, cudaChannelFormatKindUnsigned);


            // 2-channel unsigned char filtered image
            // TODO: need to check if this actually works!
            __imageFiltered = GPUImage(height, width, 2, 1);
            __imageFilteredTexture = GPUTexture(__imageFiltered, cudaChannelFormatKindUnsigned);

            // 1-channel float constant brightness model parameter
            __imageConstant = GPUImage(height, width, 1, 4);

            // 2-channel float constant brightness model parameter
            __imageGradient = GPUImage(height, width, 2, 4);

            __configured = true;
        }

        /**
         * \brief performs computation of brightness parameters
         */
        void ImageModel::compute() {

            if(!__configured) {
                std::cerr << "ERROR: ImageModel::compute() stage not configured." << std::endl;
            }

            startTiming();

            // prefilter


            // compute brightness parameters

            stopTiming();
        }


        //#########################
        // Pipeline stage inputs
        //#########################
        void ImageModel::setInputImage(flowfilter::gpu::GPUImage img) {

            // check if image is a gray scale image with pixels 1 byte long
            if(img.depth() != 1) throw std::exception();
            if(img.itemSize() != 1) throw std::exception();

            __inputImage = img;
        }

        //#########################
        // Pipeline stage outputs
        //#########################
        flowfilter::gpu::GPUImage ImageModel::getImageConstantDevice() {

            return __imageConstant;
        }

        flowfilter::gpu::GPUImage ImageModel::getImageGradientDevice() {

            return __imageGradient;
        }


    }; // namespace gpu
}; // namespace flowfilter