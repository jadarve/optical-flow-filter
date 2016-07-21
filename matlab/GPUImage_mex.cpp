/**
 * \file GPUImage_mex.cpp
 * \brief Matlab mex interface to flowfilter::gpu::GPUImage class.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <string>
#include <vector>

#include "mex.h"
#include "classHandle.h"
#include "imgutil.h"

#include "flowfilter/image.h"
#include "flowfilter/gpu/image.h"


using namespace std;
using namespace flowfilter;
using namespace flowfilter::gpu;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // Get the command string
    char cmd_buffer[64];
    if (nrhs < 1 || mxGetString(prhs[0], cmd_buffer, sizeof(cmd_buffer))) {
        mexErrMsgTxt("First input should be a command string less than 64 characters long");
    }

    string cmd(&cmd_buffer[0]);
    // mexPrintf("Input command: %s\n", cmd.c_str());

    // CONSTRUCTOR
    if (cmd == "new") {

        // Check parameters
        if (nlhs != 1) {
            mexErrMsgTxt("New: Expecting class handle as output parameter.");
        }

        if(nrhs != 5) {
            mexErrMsgTxt("[height, width, depth, itemSize] parameters are required to create GPUImage");
        }

        int height = mxGetScalar(prhs[1]);
        int width = mxGetScalar(prhs[2]);
        int depth = mxGetScalar(prhs[3]);
        int itemSize = mxGetScalar(prhs[4]);

        // Return a handle to a new C++ instance
        GPUImage* img = new GPUImage(height, width, depth, itemSize);
        // mexPrintf("img: [%d, %d, %d]: pitch: %d \titemSize: %d",
        //     img->height(), img->width(), img->depth(), img->pitch(), img->itemSize());
        plhs[0] = convertPtr2Mat<GPUImage>(img);


        // upon successful creation of object, lock the mex file until the
        // object is destroyed
        mexLock();

        return;
    }

    // Check there is a second input, which should be the class instance handle
    if (nrhs < 2) {
        mexErrMsgTxt("Second input should be a class instance handle.");
    }

    // DESTRUCTOR
    if(cmd == "delete") {
        // Destroy the C++ object
        destroyObject<GPUImage>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2) {
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        }

        // unlock the mex file. This does not check for errors while destroying the object.
        mexUnlock();
        return;
    }

    // Get the class instance pointer from the second input
    GPUImage *instance = convertMat2Ptr<GPUImage>(prhs[1]);


    // ************************************************************** //
    //                                                                //
    //                          CLASS METHODS                         //
    //                                                                //
    // * height()                                                     //
    // * width()                                                      //
    // * depth()                                                      //
    // * pitch()                                                      //
    // * itemSize()                                                   //
    // * upload()                                                     //
    // * download()                                                   //
    // * copyFrom()    TODO                                           //
    // * clear()                                                      //
    //                         OTHER OPERATIONS                       //
    // * testTextureCreation()                                        //
    // ************************************************************** //
    
    // mexPrintf("[nlhs, nrhs]: [%d, %d]", nlhs, nrhs);

    if(cmd == "height") {
        if(nlhs != 1) mexErrMsgTxt("GPUImage.height(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("GPUImage.height(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->height());
        return;
    }

    if(cmd == "width") {
        if(nlhs != 1) mexErrMsgTxt("GPUImage.width(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("GPUImage.width(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->width());
        return;
    }

    if(cmd == "depth") {
        if(nlhs != 1) mexErrMsgTxt("GPUImage.depth(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("GPUImage.depth(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->depth());
        return;
    }

    if(cmd == "pitch") {
        if(nlhs != 1) mexErrMsgTxt("GPUImage.pitch(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("GPUImage.pitch(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->pitch());
        return;
    }

    if(cmd == "itemSize") {
        if(nlhs != 1) mexErrMsgTxt("GPUImage.itemSize(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("GPUImage.itemSize(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->itemSize());
        return;
    }


    if(cmd == "upload") {
        if(nlhs != 0) mexErrMsgTxt("GPUImage.upload(): expecting zero output parameters.");
        if(nrhs != 3) mexErrMsgTxt("GPUImage.upload(): expecting 3 input parameters.");

        const mxArray* img = prhs[2];

        image_t img_w = wrapMxImage(img);
        // printImageWrappedInfo("image parameter", img_w);

        if(instance->compareShape(img_w)) {
            // upload img_w to GPU memory space
            instance->upload(img_w);    
        } else {
            mexPrintf("ERROR: GPUImage.upload(): shape missmatch between GPU image and input parameter\n");
            printGPUImageInfo("GPU image", *instance);
            printImageWrappedInfo("image parameter", img_w);
            mexErrMsgTxt("GPUImage.upload(): input image shape does not match GPU image.");
        }

        return;
    }


    if(cmd == "download") {

        if(nlhs != 1) mexErrMsgTxt("GPUImage.download(): expecting 1 output parameters.");
        if(nrhs != 2) mexErrMsgTxt("GPUImage.download(): expecting 2 input parameters.");

        mxArray* img = createMxArray(*instance);
        image_t img_w = wrapMxImage(img);

        // set img as output value
        plhs[0] = img;

        // printImageWrappedInfo("GPUImage.download()", img_w);

        if(instance->compareShape(img_w)) {
            // download img_w to GPU memory space
            instance->download(img_w);    
        } else {
            mexPrintf("ERROR: GPUImage.download(): shape missmatch between GPU image and input parameter\n");
            printGPUImageInfo("GPU image", *instance);
            printImageWrappedInfo("image parameter", img_w);
            mexErrMsgTxt("GPUImage.download(): input image shape does not match GPU image.");
        }

        return;
    }


    if(cmd == "clear") {
        if(nlhs != 0) mexErrMsgTxt("GPUImage.clear(): expecting zero output parameters.");
        if(nrhs != 2) mexErrMsgTxt("GPUImage.clear(): expecting 2 input parameters.");

        instance->clear();
        return;
    }

    if(cmd == "testTextureCreation") {

        GPUImage img = *instance;
        cudaChannelFormatKind channelFormat = cudaChannelFormatKindNone;
        if(img.itemSize() == 4) {
            mexPrintf("channel Format: float\n");
            channelFormat = cudaChannelFormatKindFloat;
        } else {
            mexPrintf("channel Format: unsigned\n");
            channelFormat = cudaChannelFormatKindUnsigned;
        }

        mexPrintf("creating GPU texture object\n");
        GPUTexture* tex = new GPUTexture(img, channelFormat);
        mexPrintf("deleting GPU texture object\n");

        return;
    }

    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized: ");
}
