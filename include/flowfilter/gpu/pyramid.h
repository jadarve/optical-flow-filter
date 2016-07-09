/**
 * \file pyramid.h
 * \brief Classes for computing image pyramids.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_PYRAMID_H_
#define FLOWFILTER_GPU_PYRAMID_H_

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/osconfig.h"
#include "flowfilter/gpu/image.h"
#include "flowfilter/gpu/pipeline.h"


namespace flowfilter {
namespace gpu {

/**
 * \brief Objects of this class compute image pyramids
 */
class FLOWFILTER_API ImagePyramid : public Stage {

public:
    ImagePyramid();
    ImagePyramid(flowfilter::gpu::GPUImage image, const int levels);
    ~ImagePyramid();

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
     * \brief perform computation
     */
    void compute();


    //#########################
    // Stage inputs
    //#########################
    void setInputImage(flowfilter::gpu::GPUImage img);
    void setLevels(const int levels);


    //#########################
    // Stage outputs
    //#########################
    flowfilter::gpu::GPUImage getImage(int level);
    int getLevels() const;


private:
    bool __configured;
    bool __inputImageSet;

    int __levels;

    flowfilter::gpu::GPUImage __inputImage;
    flowfilter::gpu::GPUTexture __inputImageTexture;

    /** Downsampled images in X */
    std::vector<flowfilter::gpu::GPUImage> __pyramidX;
    std::vector<flowfilter::gpu::GPUTexture> __pyramidTextureX;
    std::vector<dim3> __gridX;

    /** Downsampled images in Y */
    std::vector<flowfilter::gpu::GPUImage> __pyramidY;
    std::vector<flowfilter::gpu::GPUTexture> __pyramidTextureY;
    std::vector<dim3> __gridY;

    dim3 __block;

};


}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_PYRAMID_H_