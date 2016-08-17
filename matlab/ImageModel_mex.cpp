/**
 * \file FlowFilter_mex.cpp
 * \brief Matlab mex interface to flowfilter::gpu::FlowFilter class.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <string>
#include <vector>

#include "mex.h"
#include "classHandle.h"
#include "imgutil.h"

#include "flowfilter/image.h"
#include "flowfilter/gpu/imagemodel.h"


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

        if(nrhs != 2) {
            mexErrMsgTxt("[GPUImage] parameter is required to create ImageModel");
        }

        GPUImage* inputImage = convertMat2Ptr<GPUImage>(prhs[1]);

        // Return a handle to a new C++ instance
        ImageModel* filter = new ImageModel(*inputImage);
        plhs[0] = convertPtr2Mat<ImageModel>(filter);

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

        mexPrintf("ImageModel.~ImageModel()\n");
        // Destroy the C++ object
        destroyObject<ImageModel>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2) {
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        }

        // unlock the mex file. This does not check for errors while destroying the object.
        mexUnlock();
        return;
    }

    // Get the class instance pointer from the second input
    ImageModel *instance = convertMat2Ptr<ImageModel>(prhs[1]);


    // ************************************************************** //
    //                                                                //
    //                          CLASS METHODS                         //
    //                                                                //
    // * configure()                                                  //
    // * compute()                                                    //
    // * elapsedTime()                                                //
    // * setInputImage()                                              //
    // * getImageConstant()                                           //
    // * getImageGradient()                                           //
    // ************************************************************** //
    
    // mexPrintf("[nlhs, nrhs]: [%d, %d]", nlhs, nrhs);

    if(cmd == "compute") {
        if(nlhs != 0) mexErrMsgTxt("ImageModel.compute(): expecting zero output parameters.");
        if(nrhs != 2) mexErrMsgTxt("ImageModel.compute(): expecting 2 input parameters.");

        instance->compute();
        return;
    }
    
    if(cmd == "elapsedTime") {
        if(nlhs != 1) mexErrMsgTxt("ImageModel.elapsedTime(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("ImageModel.elapsedTime(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->elapsedTime());
        return;
    }

    if(cmd == "configure") {
        if(nlhs != 0) mexErrMsgTxt("ImageModel.configure(): expecting zero output parameters.");
        if(nrhs != 2) mexErrMsgTxt("ImageModel.configure(): expecting 2 input parameters.");

        instance->configure();
        return;
    }

    if(cmd == "setInputImage") {
        
        if(nlhs != 0) mexErrMsgTxt("ImageModel.elapsedTime(): expecting 1 output parameter.");
        if(nrhs != 3) mexErrMsgTxt("ImageModel.elapsedTime(): expecting 3 input parameters.");

        GPUImage* inputImage = convertMat2Ptr<GPUImage>(prhs[2]);
        instance->setInputImage(*inputImage);

        return;
    }


    if(cmd == "getImageConstant") {
        
        if(nlhs != 0) mexErrMsgTxt("ImageModel.getImageConstant(): expecting 1 output parameter.");
        if(nrhs != 3) mexErrMsgTxt("ImageModel.getImageConstant(): expecting 3 input parameters.");

        GPUImage* imgConstant = convertMat2Ptr<GPUImage>(prhs[2]);
        // GPUImage* inputImage = convertMat2Ptr<GPUImage>(prhs[1]);

        // replace memory buffer of imgConstant by the buffer allocated by
        // instance. Previous memory buffer of imgConstant is released automatically.
        *imgConstant = instance->getImageConstant();

        return;
    }


    if(cmd == "getImageGradient") {
        
        if(nlhs != 0) mexErrMsgTxt("ImageModel.getImageGradient(): expecting 1 output parameter.");
        if(nrhs != 3) mexErrMsgTxt("ImageModel.getImageGradient(): expecting 3 input parameters.");

        GPUImage* imgGradient = convertMat2Ptr<GPUImage>(prhs[2]);

        // replace memory buffer of imgGradient by the buffer allocated by
        // instance. Previous memory buffer of imgGradient is released automatically.
        *imgGradient = instance->getImageGradient();

        return;
    }




    // if(cmd == "downloadFlow") {
    //     if(nlhs != 1) mexErrMsgTxt("GPUImage.downloadFlow(): expecting 1 output parameters.");
    //     if(nrhs != 2) mexErrMsgTxt("GPUImage.downloadFlow(): expecting 2 input parameters.");

    //     // create uint8 image
    //     mxArray* flow = createMxArray(instance->height(), instance->width(), 2, mxSINGLE_CLASS);
    //     image_t flow_w = wrapMxImage(flow);

    //     // set flow as output value
    //     plhs[0] = flow;
    //     instance->downloadFlow(flow_w);

    //     // float* ptr = (float*)mxGetData(flow);
    //     // // print first elements
    //     // for(unsigned int i = 0; i < 10; i ++) {
    //     //     mexPrintf("i: %d\t=%f\n", i, ptr[i]);
    //     // }

    //     return;
    // }

    // if(cmd == "downloadImage") {
    //     if(nlhs != 1) mexErrMsgTxt("GPUImage.downloadImage(): expecting 1 output parameters.");
    //     if(nrhs != 2) mexErrMsgTxt("GPUImage.downloadImage(): expecting 2 input parameters.");

    //     // create uint8 image
    //     mxArray* img = createMxArray(instance->height(), instance->width(), mxSINGLE_CLASS);
    //     image_t img_w = wrapMxImage(img);

    //     // set img as output value
    //     plhs[0] = img;
    //     instance->downloadImage(img_w);

    //     return;
    // }

    // if(cmd == "setGamma") {
    //     if(nlhs != 0) mexErrMsgTxt("FlowFilter.setGamma(): expecting zero output parameter.");
    //     if(nrhs != 3) mexErrMsgTxt("FlowFilter.setGamma(): expecting 3 input parameters.");

    //     instance->setGamma(mxGetScalar(prhs[2]));
    //     return;
    // }

    // if(cmd == "getGamma") {
    //     if(nlhs != 1) mexErrMsgTxt("FlowFilter.getGamma(): expecting 1 output parameter.");
    //     if(nrhs != 2) mexErrMsgTxt("FlowFilter.getGamma(): expecting 2 input parameters.");

    //     plhs[0] = mxCreateDoubleScalar(instance->getGamma());
    //     return;
    // }

    // if(cmd == "setMaxFlow") {
    //     if(nlhs != 0) mexErrMsgTxt("FlowFilter.setMaxFlow(): expecting zero output parameter.");
    //     if(nrhs != 3) mexErrMsgTxt("FlowFilter.setMaxFlow(): expecting 3 input parameters.");

    //     instance->setMaxFlow(mxGetScalar(prhs[2]));
    //     return;
    // }

    // if(cmd == "getMaxFlow") {
    //     if(nlhs != 1) mexErrMsgTxt("FlowFilter.getMaxFlow(): expecting 1 output parameter.");
    //     if(nrhs != 2) mexErrMsgTxt("FlowFilter.getMaxFlow(): expecting 2 input parameters.");

    //     plhs[0] = mxCreateDoubleScalar(instance->getMaxFlow());
    //     return;
    // }

    // if(cmd == "setSmoothIterations") {
    //     if(nlhs != 0) mexErrMsgTxt("FlowFilter.setSmoothIterations(): expecting zero output parameter.");
    //     if(nrhs != 3) mexErrMsgTxt("FlowFilter.setSmoothIterations(): expecting 3 input parameters.");

    //     instance->setSmoothIterations((int)mxGetScalar(prhs[2]));
    //     return;
    // }

    // if(cmd == "getSmoothIterations") {
    //     if(nlhs != 1) mexErrMsgTxt("FlowFilter.getSmoothIterations(): expecting 1 output parameter.");
    //     if(nrhs != 2) mexErrMsgTxt("FlowFilter.getSmoothIterations(): expecting 2 input parameters.");

    //     plhs[0] = mxCreateDoubleScalar(instance->getSmoothIterations());
    //     return;
    // }

    // if(cmd == "setPropagationBorder") {
    //     if(nlhs != 0) mexErrMsgTxt("FlowFilter.setPropagationBorder(): expecting zero output parameter.");
    //     if(nrhs != 3) mexErrMsgTxt("FlowFilter.setPropagationBorder(): expecting 3 input parameters.");

    //     instance->setPropagationBorder((int)mxGetScalar(prhs[2]));
    //     return;
    // }

    // if(cmd == "getPropagationBorder") {
    //     if(nlhs != 1) mexErrMsgTxt("FlowFilter.getPropagationBorder(): expecting 1 output parameter.");
    //     if(nrhs != 2) mexErrMsgTxt("FlowFilter.getPropagationBorder(): expecting 2 input parameters.");

    //     plhs[0] = mxCreateDoubleScalar(instance->getPropagationBorder());
    //     return;
    // }

    // if(cmd == "getPropagationIterations") {
    //     if(nlhs != 1) mexErrMsgTxt("FlowFilter.getPropagationIterations(): expecting 1 output parameter.");
    //     if(nrhs != 2) mexErrMsgTxt("FlowFilter.getPropagationIterations(): expecting 2 input parameters.");

    //     plhs[0] = mxCreateDoubleScalar(instance->getPropagationIterations());
    //     return;
    // }

    // if(cmd == "configure") {
    //     if(nlhs != 0) mexErrMsgTxt("FlowFilter.configure(): expecting zero output parameters.");
    //     if(nrhs != 2) mexErrMsgTxt("FlowFilter.configure(): expecting 2 input parameters.");

    //     instance->configure();
    //     return;
    // }

    // if(cmd == "height") {
    //     if(nlhs != 1) mexErrMsgTxt("FlowFilter.height(): expecting 1 output parameter.");
    //     if(nrhs != 2) mexErrMsgTxt("FlowFilter.height(): expecting 2 input parameters.");

    //     plhs[0] = mxCreateDoubleScalar(instance->height());
    //     return;
    // }

    // if(cmd == "width") {
    //     if(nlhs != 1) mexErrMsgTxt("FlowFilter.width(): expecting 1 output parameter.");
    //     if(nrhs != 2) mexErrMsgTxt("FlowFilter.width(): expecting 2 input parameters.");

    //     plhs[0] = mxCreateDoubleScalar(instance->width());
    //     return;
    // }

    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized: ");
}
