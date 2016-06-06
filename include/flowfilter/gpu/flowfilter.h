/**
 * \file flowfilter.h
 * \brief Optical flow filter classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_FLOWFILTER_H_
#define FLOWFILTER_GPU_FLOWFILTER_H_

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/osconfig.h"
#include "flowfilter/image.h"

#include "flowfilter/gpu/image.h"
#include "flowfilter/gpu/pipeline.h"
#include "flowfilter/gpu/imagemodel.h"
#include "flowfilter/gpu/update.h"
#include "flowfilter/gpu/propagation.h"
#include "flowfilter/gpu/flowsmoothing.h"
#include "flowfilter/gpu/pyramid.h"


namespace flowfilter {
namespace gpu {

class FLOWFILTER_API FlowFilter : public Stage {

public:
    FlowFilter();
    FlowFilter(flowfilter::gpu::GPUImage inputImage);
    FlowFilter(const int height, const int witdh);
    FlowFilter(const int height, const int witdh,
        const int smoothIterations,
        const float maxflow,
        const float gamma);
    ~FlowFilter();

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

    void computeImageModel();
    void computePropagation();
    void computeUpdate();

    //#########################
    // Stage inputs
    //#########################

    void setInputImage(flowfilter::gpu::GPUImage inputImage);

    //#########################
    // Stage outputs
    //#########################

    flowfilter::gpu::GPUImage getFlow();


    //#########################
    // Host load-download
    //#########################

    /**
     * \brief load image stored in CPU memory space
     */
    void loadImage(flowfilter::image_t& image);

    /**
     * \brief returns the new estimate of optical flow
     */
    void downloadFlow(flowfilter::image_t& flow);

    /**
     * \brief returns current brightness model constant value, corresponding
     *      to a smoothed version of the original image
     */
    void downloadImage(flowfilter::image_t& image);

    // // Image model outputs
    // void downloadImageGradient(flowfilter::image_t& gradient);
    // void downloadImageConstant(flowfilter::image_t& gradient);

    // // Update stage
    // void downloadFlowUpdated(flowfilter::image_t& flow);
    // void downloadImageUpdated(flowfilter::image_t& image);

    // // Smooth stage
    // void downloadSmoothedFlow(flowfilter::image_t& flow);

    
    //#########################
    // Parameters
    //#########################

    float getGamma() const;
    void setGamma(const float gamma);

    float getMaxFlow() const;
    void setMaxFlow(const float maxflow);

    int getSmoothIterations() const;
    void setSmoothIterations(const int N);

    void setPropagationBorder(const int border);
    int getPropagationBorder() const;

    int getPropagationIterations() const;

    int height() const;
    int width() const;


private:
    int __height;
    int __width;

    bool __configured;
    bool __firstLoad;
    bool __inputImageSet;

    flowfilter::gpu::GPUImage __inputImage;

    flowfilter::gpu::ImageModel __imageModel;
    flowfilter::gpu::FlowUpdate __update;
    flowfilter::gpu::FlowSmoother __smoother;
    flowfilter::gpu::FlowPropagator __propagator;

};


class FLOWFILTER_API DeltaFlowFilter : public Stage {

public:
    DeltaFlowFilter();
    DeltaFlowFilter(flowfilter::gpu::GPUImage inputImage,
        flowfilter::gpu::GPUImage inputFlow);
    ~DeltaFlowFilter();

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


    void computeImageModel();
    void computePropagation();
    void computeUpdate();

    //#########################
    // Stage inputs
    //#########################

    void setInputImage(flowfilter::gpu::GPUImage inputImage);
    void setInputFlow(flowfilter::gpu::GPUImage inputFlow);


    //#########################
    // Stage outputs
    //#########################

    flowfilter::gpu::GPUImage getFlow();
    flowfilter::gpu::GPUImage getImage();


    //#########################
    // Parameters
    //#########################

    float getGamma() const;
    void setGamma(const float gamma);

    float getMaxFlow() const;
    void setMaxFlow(const float maxflow);

    int getSmoothIterations() const;
    void setSmoothIterations(const int N);

    void setPropagationBorder(const int border);
    int getPropagationBorder() const;

    int getPropagationIterations() const;

    int height() const;
    int width() const;


private:
    int __height;
    int __width;

    bool __configured;
    bool __firstLoad;

    /** Tells if __inputImage has been set from a external source */
    bool __inputImageSet;

    bool __inputFlowSet;

    flowfilter::gpu::GPUImage __inputImage;
    flowfilter::gpu::GPUImage __inputFlow;

    flowfilter::gpu::ImageModel __imageModel;
    flowfilter::gpu::DeltaFlowUpdate __update;
    flowfilter::gpu::FlowSmoother __smoother;
    flowfilter::gpu::FlowPropagatorPayload __propagator;

};


class FLOWFILTER_API PyramidalFlowFilter : public Stage {

public:
    PyramidalFlowFilter();
    PyramidalFlowFilter(const int height, const int width, const int levels);
    ~PyramidalFlowFilter();


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
    // Stage outputs
    //#########################

    flowfilter::gpu::GPUImage getFlow();


    //#########################
    // Host load-download
    //#########################

    /**
     * \brief load image stored in CPU memory space
     */
    void loadImage(flowfilter::image_t& image);

    /**
     * \brief returns the new estimate of optical flow
     */
    void downloadFlow(flowfilter::image_t& flow);

    /**
     * \brief returns current brightness model constant value, corresponding
     *      to a smoothed version of the original image
     */
    void downloadImage(flowfilter::image_t& image);


    //#########################
    // Parameters
    //#########################

    float getGamma(const int level) const;
    void setGamma(const int level, const float gamma);
    void setGamma(const std::vector<float>& gamma);

    float getMaxFlow() const;
    void setMaxFlow(const float maxflow);

    int getSmoothIterations(const int level) const;
    void setSmoothIterations(const int level, const int N);
    void setSmoothIterations(const std::vector<int>& iterations);

    void setPropagationBorder(const int border);
    int getPropagationBorder() const;
    
    // int getPropagationIterations() const;

    int height() const;
    int width() const;
    int levels() const;


private:

    bool __configured;

    int __height;
    int __width;
    int __levels;

    flowfilter::gpu::GPUImage __inputImage;

    flowfilter::gpu::ImagePyramid __imagePyramid;

    flowfilter::gpu::FlowFilter __topLevelFilter;

    std::vector<DeltaFlowFilter> __lowLevelFilters;

};

}; // namespace gpu
}; // namespace flowfilter

#endif // FLOWFILTER_GPU_FLOWFILTER_H_