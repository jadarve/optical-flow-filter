/**
 * \file flowWebCam.cpp
 * \brief Optical flow demo using OpenCV VideoCapture to compute flow from webcam.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <vector>

#include <ctime>
#include <unistd.h>
#include <sys/time.h>

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

    int cameraIndex = 0;
    timeval start, end;

    // if user provides camera index
    if(argc > 1) {
        cameraIndex = atoi(argv[1]);
    }

    VideoCapture cap(cameraIndex); // open the default camera
    if(!cap.isOpened()){
        return -1;
    }

    Mat frame, frameTrue;

    //  captura a frame to get image width and height
    cap >> frameTrue;
    cv::flip(frameTrue, frame, 1);
    int width = frame.cols;
    int height = frame.rows;
    cout << "frame shape: [" << height << ", " << width << "]" << endl;

    Mat frameGray(height, width, CV_8UC1);
    Mat fcolor(height, width, CV_8UC4);

    // structs used to wrap cv::Mat images and transfer to flowfilter
    image_t hostImageGray;
    image_t hostFlowColor;

    wrapCVMat(frameGray, hostImageGray);
    wrapCVMat(fcolor, hostFlowColor);

    //#################################
    // Filter parameters
    //#################################
    float maxflow = 4.0f;
    vector<float> gamma = {10, 50,10};
    vector<int> smoothIterations = {3, 3,3};

    //#################################
    // Filter creation with
    // 3 pyramid levels
    //#################################
    PyramidalFlowFilter filter(height, width, 3);
    filter.setMaxFlow(maxflow);
    filter.setGamma(gamma);
    filter.setSmoothIterations(smoothIterations);

    //#################################
    // To access optical flow
    // on the host
    //#################################
    Mat flowHost(height, width, CV_32FC2);
    image_t flowHostWrapper;
    wrapCVMat(flowHost, flowHostWrapper);

    // Color encoder connected to optical flow buffer in the GPU
    FlowToColor flowColor(filter.getFlow(), maxflow, 1);


    // Capture loop
    for(;;) {
        

        // capture a new frame from the camera
        // and convert it to gray scale (uint8)
        gettimeofday(&start, 0);
        cap >> frameTrue;
        
        cv::flip(frameTrue, frame, 1);
        
        cvtColor(frame, frameGray, CV_BGR2GRAY);

        // transfer image to flow filter and compute
        
        filter.loadImage(hostImageGray);
        filter.compute();

        cout << "elapsed time: " << filter.elapsedTime() << " ms" << endl;

        // transfer the optical flow from GPU to
        // host memory allocated by flowHost cvMat.
        // After this, optical flow values
        // can be accessed using OpenCV pixel
        // access methods.
        filter.downloadFlow(flowHostWrapper);

        // computes color encoding (RGBA) and download it to host
        flowColor.compute();
        flowColor.downloadColorFlow(hostFlowColor);
        cvtColor(fcolor, fcolor, CV_RGBA2BGRA);

        imshow("image", frameGray);
        imshow("optical flow", fcolor);
	    waitKey(5);
        //if(waitKey(5) >= 0) break;
        gettimeofday(&end, 0);
        //cout << "difference: " << (end.tv_usec/1000.0 - start.tv_usec/1000.0) << endl << endl;
        cout << "fps: " <<  1000.0/(end.tv_usec/1000.0 - start.tv_usec/1000.0) << endl << endl;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
