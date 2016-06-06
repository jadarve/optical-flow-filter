/**
 * \file pipeline.h
 * \brief type declarations vision pipelines.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWFILTER_GPU_PIPELINE_H_
#define FLOWFILTER_GPU_PIPELINE_H_

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include "flowfilter/osconfig.h"

namespace flowfilter {
namespace gpu {

/**
 * \brief Abstract class
 *
 * This class exposes the basic functionality
 * of a pipeline stage. It declares methods
 * to perfom computation and evaluate runtime
 * of the stage. 
 */
class FLOWFILTER_API Stage {

public:

    /**
     * \brief creates a pipeline stage on the default stream
     */
    Stage();

    /**
     * \brief creates a pipeline stage on a given CUDA stream
     */
    Stage(cudaStream_t stream);
    virtual ~Stage();

    void startTiming();
    void stopTiming();

    /**
     * \brief configures the stage.
     *
     * After configuration, calls to compute()
     * are valid.
     * Input buffers should not change after
     * this method has been called.
     */
    virtual void configure() = 0;

    /**
     * \brief perform computation
     */
    virtual void compute() = 0;

    /**
     * \brief return computation elapsed time in milliseconds
     */
    float elapsedTime() const;


protected:
    /** CUDA stream to which this stage belongs */
    cudaStream_t __stream;

private:
    cudaEvent_t __start;
    cudaEvent_t __stop;
    float __elapsedTime;

    std::shared_ptr<int> __referenceCounter;
};


/**
 * \brief Pipeline stage with empty compute() implementation
 *
 * Implementation of the compute method only calls
 * startTiming() and stopTiming().
 */
class FLOWFILTER_API EmptyStage : public Stage {

public:
    EmptyStage();
    ~EmptyStage();

    void configure();
    void compute();
};

};
};
#endif // FLOWFILTER_GPU_PIPELINE_H_