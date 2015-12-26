/**
 * \file pipeline.cu
 * \brief type declarations vision pipelines.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <exception>
#include <iostream>

#include "flowfilter/gpu/error.h"
#include "flowfilter/gpu/pipeline.h"


namespace flowfilter {
    namespace gpu {

        //#################################################
        // Stage
        //#################################################
        Stage::Stage() :
            Stage(0) {
        }

        Stage::Stage(cudaStream_t stream) {
            checkError(cudaSetDevice(0));
            __stream = stream;
            __elapsedTime = 0.0f;
            __referenceCounter = std::make_shared<int>(0);

            checkError(cudaEventCreate(&__start));
            checkError(cudaEventCreate(&__stop));

            // if(startErr != cudaSuccess || stopErr != cudaSuccess) {
            //     std::cerr << "Stage::Stage(): error creating CUDA events: "
            //     << cudaGetErrorString(startErr) << " - "
            //     << cudaGetErrorString(stopErr) << std::endl;

            //     throw std::exception();
            // }
        }

        Stage::~Stage() {

            // std::cout << "Stage::~Stage(): " << __referenceCounter.use_count() << std::endl;

            if(__referenceCounter.use_count() == 1) {

                checkError(cudaEventDestroy(__start));
                checkError(cudaEventDestroy(__stop));

                // if(startErr != cudaSuccess || stopErr != cudaSuccess) {
                //     std::cerr << "Stage::Stage(): error destroying CUDA events: "
                //     << cudaGetErrorString(startErr) << " - "
                //     << cudaGetErrorString(stopErr) << std::endl;

                //     throw std::exception();
                // }    
            }
        }


        void Stage::startTiming() {
            checkError(cudaEventRecord(__start, __stream));
            // cudaError_t startErr = cudaGetLastError();
            // if(startErr != cudaSuccess) {
            //     std::cerr << "ERROR: Stage::startTiming(): error starting timing: "
            //     << cudaGetErrorString(startErr) << std::endl;
            // }
        }

        void Stage::stopTiming() {
            checkError(cudaEventRecord(__stop, __stream));
            checkError(cudaEventSynchronize(__stop));
            checkError(cudaEventElapsedTime(&__elapsedTime, __start, __stop));

            cudaError_t stopErr = cudaGetLastError();
            if(stopErr != cudaSuccess) {
                std::cerr << "ERROR: Stage::startTiming(): error stoping timing: "
                << cudaGetErrorString(stopErr) << std::endl;
            }
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