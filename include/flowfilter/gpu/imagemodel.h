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

#include "flowfilter/osconfig.h"
#include "flowfilter/gpu/pipeline.h"
#include "flowfilter/gpu/image.h"

namespace flowfilter {
namespace gpu {


class FLOWFILTER_API ImageModel : public Stage {

public:
    ImageModel();

    /**
     * \brief creates an image model stage with a given input image
     *
     * This constructor internally calles configure() so that the
     * stage is ready to perform computations.
     */
    ImageModel(flowfilter::gpu::GPUImage inputImage);

    ~ImageModel();

public:

    /**
     * \brief configures the stage.
     *
     * After configuration, calls to compute()
     * are valid.
     * Input buffers should not change after
     * this method has been called.
     */
    void configure();

    /**
     * \brief performs computation of brightness parameters
     */
    void compute();

    //#########################
    // Stage inputs
    //#########################
    void setInputImage(flowfilter::gpu::GPUImage img);

    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getImageConstant();
    flowfilter::gpu::GPUImage getImageGradient();

private:

    // tell if the stage has been configured
    bool __configured;

    /** tells if an input image has been set */
    bool __inputImageSet;

    // inputs
    flowfilter::gpu::GPUImage __inputImage;
    flowfilter::gpu::GPUTexture __inputImageTexture;

    // outputs
    flowfilter::gpu::GPUImage __imageConstant;
    flowfilter::gpu::GPUImage __imageGradient;

    // intermediate buffers

    /** 2-channels image with X and Y filtering version of inputImage */
    flowfilter::gpu::GPUImage __imageFiltered;
    flowfilter::gpu::GPUTexture __imageFilteredTexture;

    // block and grid size for kernel calls
    dim3 __block;
    dim3 __grid;

};

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_IMAGEMODEL_H_