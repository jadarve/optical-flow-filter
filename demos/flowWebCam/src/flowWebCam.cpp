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


int main(int argc, char** argv) {

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened()){
        return -1;
    }
    
    Mat frame;
    

    //  captura a frame to get image width and height
    cap >> frame;
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
    float maxflow = 20.0f;
    vector<float> gamma = {500.0f, 50.0f, 5.0f};
    vector<int> smoothIterations = {2, 8, 20};

    //#################################
    // Filter creation with
    // 3 pyramid levels
    //#################################
    PyramidalFlowFilter filter(height, width, 3);
    filter.setMaxFlow(maxflow);
    filter.setGamma(gamma);
    filter.setSmoothIterations(smoothIterations);
    

    // Color encoder connected to optical flow buffer in the GPU
    FlowToColor flowColor(filter.getFlow(), maxflow);


    // Capture loop
    for(;;) {

        // capture a new frame from the camera
        // and convert it to gray scale (uint8)
        cap >> frame;
        cvtColor(frame, frameGray, CV_BGR2GRAY);
        
        // transfer image to flow filter and compute
        filter.loadImage(hostImageGray);
        filter.compute();

        cout << "elapsed time: " << filter.elapsedTime() << " ms" << endl;

        // computes color encoding (RGBA) and download it to host
        flowColor.compute();
        flowColor.downloadColorFlow(hostFlowColor);
        cvtColor(fcolor, fcolor, CV_RGBA2BGRA);

        imshow("image", frameGray);
        imshow("optical flow", fcolor);

        if(waitKey(1) >= 0) break;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}