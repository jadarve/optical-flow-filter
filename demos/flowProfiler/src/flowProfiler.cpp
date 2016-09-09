/**
 * \file flowWebCam.cpp
 * \brief Optical flow demo using OpenCV VideoCapture to compute flow from webcam.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <flowfilter/gpu/flowfilter.h>
#include <flowfilter/gpu/display.h>

using namespace std;
using namespace cv;
using namespace flowfilter;
using namespace flowfilter::gpu;


void wrapCVMat(Mat& cvMat, image_t& img) {

    img.height = cvMat.rows;
    img.width = cvMat.cols;
    img.depth = cvMat.channels();
    img.pitch = cvMat.cols*cvMat.elemSize();
    img.itemSize = cvMat.elemSize1();
    img.data = cvMat.ptr();
}

/**
 * MODE OF USE
 * ./flowWebCam <cameraIndex>
 *
 * where <cameraIndex> is an integer indicating the camera used
 * to capture images. Defaults to 0;
 *
 */
int main(int argc, char** argv) {

    if(argc < 6) {
        cerr << "ERROR: expecting 4 arguments: height, width, pyrLevels, maxFlow, iterations" << endl;
        return -1;
    }
    

    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    int pyrLevels = atoi(argv[3]);
    int maxFlow_i = atoi(argv[4]);
    int N = atoi(argv[5]);
    

    cout << "image shape: [" << height << ", " << width << "]" << endl;
    cout << "pyramid levels: " << pyrLevels << endl;
    cout << "max flow (pixels): " << maxFlow_i << endl;

    

    //#################################
    // Filter parameters
    //#################################
    float maxflow = (float)maxFlow_i;
    
    vector<float> gamma(pyrLevels);
    vector<int> smoothIterations(pyrLevels);

    for(int i = 0; i < pyrLevels; i ++) {
        gamma[i] = 1.0f;
        smoothIterations[i] = 1;
    }

    //#################################
    // Filter creation with
    // 3 pyramid levels
    //#################################
    PyramidalFlowFilter filter(height, width, pyrLevels);
    filter.setMaxFlow(maxflow);
    filter.setGamma(gamma);
    filter.setSmoothIterations(smoothIterations);

    
    // host image
    Mat hostImage(height, width, CV_8UC1);
    image_t hostImageWrapped;
    wrapCVMat(hostImage, hostImageWrapped);
    
    // Capture loop
    for(int i = 0; i < N; i ++) {

        
        // transfer image to flow filter and compute
        filter.loadImage(hostImageWrapped);
        filter.compute();

        // cout << "elapsed time: " << filter.elapsedTime() << " ms" << endl;
        cout << filter.elapsedTime() << endl;

        // transfer the optical flow from GPU to
        // host memory allocated by flowHost cvMat.
        // After this, optical flow values
        // can be accessed using OpenCV pixel
        // access methods.
        // filter.downloadFlow(flowHostWrapper);
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
