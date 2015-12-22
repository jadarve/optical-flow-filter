/**
 * \file pipeline.cu
 * \brief type declarations vision pipelines.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#include "flowfilter/gpu/pipeline.h"

#include <iostream>

namespace flowfilter {
    namespace gpu {

        //#################################################
        // Stage
        //#################################################
        Stage::Stage() :
            Stage(0) {
        }

        Stage::Stage(cudaStream_t stream) {

            __stream = stream;
            __elapsedTime = 0.0f;

            cudaError_t startErr = cudaEventCreate(&__start);
            cudaError_t stopErr = cudaEventCreate(&__stop);

            if(startErr != cudaSuccess || stopErr != cudaSuccess) {
                std::cerr << "Stage::Stage(): error creating CUDA events: "
                << cudaGetErrorString(startErr) << " - "
                << cudaGetErrorString(stopErr) << std::endl;

                // TODO: throw exception
            }
        }

        Stage::~Stage() {

            cudaError_t startErr = cudaEventDestroy(__start);
            cudaError_t stopErr = cudaEventDestroy(__stop);

            if(startErr != cudaSuccess || stopErr != cudaSuccess) {
                std::cerr << "Stage::Stage(): error destroying CUDA events: "
                << cudaGetErrorString(startErr) << " - "
                << cudaGetErrorString(stopErr) << std::endl;

                // TODO: throw exception
            }
        }


        void Stage::startTiming() {
            cudaEventRecord(__start, __stream);
        }

        void Stage::stopTiming() {
            cudaEventRecord(__stop, __stream);
            cudaEventSynchronize(__stop);
            cudaEventElapsedTime(&__elapsedTime, __start, __stop);
        }

        /**
         * \brief return computation elapsed time in milliseconds
         */
        float Stage::elapsedTime() const {
            return __elapsedTime;
        }


        //#################################################
        // EmptyStage
        //#################################################
        EmptyStage::EmptyStage() :
            Stage() {

            // nothing to do
        }

        EmptyStage::~EmptyStage() {
            // nothing to do
        }

        void EmptyStage::configure() {
            // nothing to do...
        }

        void EmptyStage::compute() {
            
            startTiming();

            // no operation to be performed

            stopTiming();
        }

    };
};