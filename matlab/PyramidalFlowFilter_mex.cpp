/**
 * \file PyramidalFlowFilter_mex.cpp
 * \brief Matlab mex interface to flowfilter::gpu::PyramidalFlowFilter class.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <string>
#include <vector>

#include "mex.h"
#include "classHandle.h"
#include "imgutil.h"

#include "flowfilter/image.h"
#include "flowfilter/gpu/flowfilter.h"


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

        if(nrhs != 4) {
            mexErrMsgTxt("[height, width, levels] parameters are required to create PyramidalFlowFilter");
        }

        int height = mxGetScalar(prhs[1]);
        int width = mxGetScalar(prhs[2]);
        int levels = mxGetScalar(prhs[3]);

        mexPrintf("PyramidalFlowFilter.new(): [%d, %d, %d]\n", height, width, levels);

        // Return a handle to a new C++ instance
        PyramidalFlowFilter* filter = new PyramidalFlowFilter(height, width, levels);
        plhs[0] = convertPtr2Mat<PyramidalFlowFilter>(filter);

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
        destroyObject<PyramidalFlowFilter>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2) {
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        }

        // unlock the mex file. This does not check for errors while destroying the object.
        mexUnlock();
        
        return;
    }

    // Get the class instance pointer from the second input
    PyramidalFlowFilter *instance = convertMat2Ptr<PyramidalFlowFilter>(prhs[1]);


    // ************************************************************** //
    //                                                                //
    //                          CLASS METHODS                         //
    //                                                                //
    // * height()                                                     //
    // * width()                                                      //
    // * configure()                                                  //
    // * compute()                                                    //
    // * elapsedTime()                                                //
    // * loadImage()                                                  //
    // * downloadImage()                                              //
    // * downloadFlow()                                               //
    // * setGamma()                                                   //
    // * getGamma()                                                   //
    // * getMaxFlow()                                                 //
    // * setMaxFlow()                                                 //
    // * getSmoothIterations()                                        //
    // * setSmoothIterations()                                        //
    // * getPropagationBorder()                                       //
    // * setPropagationBorder()                                       //
    // * getPropagationIterations()                                   //
    // ************************************************************** //
    
    // mexPrintf("[nlhs, nrhs]: [%d, %d]", nlhs, nrhs);
    
    if(cmd == "loadImage") {

        if(nlhs != 0) mexErrMsgTxt("PyramidalFlowFilter.upload(): expecting zero output parameters.");
        if(nrhs != 3) mexErrMsgTxt("PyramidalFlowFilter.upload(): expecting 3 input parameters.");

        const mxArray* img = prhs[2];

        image_t img_w = wrapMxImage(img);
        // printImageWrappedInfo("image parameter", img_w);

        instance->loadImage(img_w);
        return;
    }

    if(cmd == "compute") {
        if(nlhs != 0) mexErrMsgTxt("PyramidalFlowFilter.compute(): expecting zero output parameters.");
        if(nrhs != 2) mexErrMsgTxt("PyramidalFlowFilter.compute(): expecting 2 input parameters.");

        instance->compute();
        return;
    }
    
    if(cmd == "elapsedTime") {
        if(nlhs != 1) mexErrMsgTxt("PyramidalFlowFilter.elapsedTime(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("PyramidalFlowFilter.elapsedTime(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->elapsedTime());
        return;
    }

    if(cmd == "downloadFlow") {
        if(nlhs != 1) mexErrMsgTxt("GPUImage.downloadFlow(): expecting 1 output parameters.");
        if(nrhs != 2) mexErrMsgTxt("GPUImage.downloadFlow(): expecting 2 input parameters.");

        // create uint8 image
        mxArray* flow = createMxArray(instance->height(), instance->width(), 2, mxSINGLE_CLASS);
        image_t flow_w = wrapMxImage(flow);

        // set flow as output value
        plhs[0] = flow;
        instance->downloadFlow(flow_w);

        return;
    }

    if(cmd == "downloadImage") {
        if(nlhs != 1) mexErrMsgTxt("GPUImage.downloadImage(): expecting 1 output parameters.");
        if(nrhs != 2) mexErrMsgTxt("GPUImage.downloadImage(): expecting 2 input parameters.");

        // create uint8 image
        mxArray* img = createMxArray(instance->height(), instance->width(), mxSINGLE_CLASS);
        image_t img_w = wrapMxImage(img);

        // set img as output value
        plhs[0] = img;
        instance->downloadImage(img_w);

        return;
    }

    if(cmd == "setGamma") {
        if(nlhs != 0) mexErrMsgTxt("PyramidalFlowFilter.setGamma(): expecting zero output parameter.");
        if(nrhs != 4) mexErrMsgTxt("PyramidalFlowFilter.setGamma(): expecting 4 input parameters.");

        instance->setGamma((int)mxGetScalar(prhs[2]), mxGetScalar(prhs[3]));
        return;
    }

    if(cmd == "getGamma") {
        if(nlhs != 1) mexErrMsgTxt("PyramidalFlowFilter.getGamma(): expecting 1 output parameter.");
        if(nrhs != 3) mexErrMsgTxt("PyramidalFlowFilter.getGamma(): expecting 3 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->getGamma((int)mxGetScalar(prhs[2])));
        return;
    }

    if(cmd == "setMaxFlow") {
        if(nlhs != 0) mexErrMsgTxt("PyramidalFlowFilter.setMaxFlow(): expecting zero output parameter.");
        if(nrhs != 3) mexErrMsgTxt("PyramidalFlowFilter.setMaxFlow(): expecting 3 input parameters.");

        instance->setMaxFlow(mxGetScalar(prhs[2]));
        return;
    }

    if(cmd == "getMaxFlow") {
        if(nlhs != 1) mexErrMsgTxt("PyramidalFlowFilter.getMaxFlow(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("PyramidalFlowFilter.getMaxFlow(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->getMaxFlow());
        return;
    }

    if(cmd == "setSmoothIterations") {
        if(nlhs != 0) mexErrMsgTxt("PyramidalFlowFilter.setSmoothIterations(): expecting zero output parameter.");
        if(nrhs != 4) mexErrMsgTxt("PyramidalFlowFilter.setSmoothIterations(): expecting 4 input parameters.");

        instance->setSmoothIterations((int)mxGetScalar(prhs[2]), mxGetScalar(prhs[3]));
        return;
    }

    if(cmd == "getSmoothIterations") {
        if(nlhs != 1) mexErrMsgTxt("PyramidalFlowFilter.getSmoothIterations(): expecting 1 output parameter.");
        if(nrhs != 3) mexErrMsgTxt("PyramidalFlowFilter.getSmoothIterations(): expecting 3 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->getSmoothIterations((int)mxGetScalar(prhs[2])));
        return;
    }

    if(cmd == "setPropagationBorder") {
        if(nlhs != 0) mexErrMsgTxt("PyramidalFlowFilter.setPropagationBorder(): expecting zero output parameter.");
        if(nrhs != 3) mexErrMsgTxt("PyramidalFlowFilter.setPropagationBorder(): expecting 3 input parameters.");

        instance->setPropagationBorder((int)mxGetScalar(prhs[2]));
        return;
    }

    if(cmd == "getPropagationBorder") {
        if(nlhs != 1) mexErrMsgTxt("PyramidalFlowFilter.getPropagationBorder(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("PyramidalFlowFilter.getPropagationBorder(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->getPropagationBorder());
        return;
    }

    // if(cmd == "getPropagationIterations") {
    //     if(nlhs != 1) mexErrMsgTxt("PyramidalFlowFilter.getPropagationIterations(): expecting 1 output parameter.");
    //     if(nrhs != 2) mexErrMsgTxt("PyramidalFlowFilter.getPropagationIterations(): expecting 2 input parameters.");

    //     plhs[0] = mxCreateDoubleScalar(instance->getPropagationIterations());
    //     return;
    // }

    if(cmd == "configure") {
        if(nlhs != 0) mexErrMsgTxt("PyramidalFlowFilter.configure(): expecting zero output parameters.");
        if(nrhs != 2) mexErrMsgTxt("PyramidalFlowFilter.configure(): expecting 2 input parameters.");

        instance->configure();
        return;
    }

    if(cmd == "height") {
        if(nlhs != 1) mexErrMsgTxt("PyramidalFlowFilter.height(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("PyramidalFlowFilter.height(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->height());
        return;
    }

    if(cmd == "width") {
        if(nlhs != 1) mexErrMsgTxt("PyramidalFlowFilter.width(): expecting 1 output parameter.");
        if(nrhs != 2) mexErrMsgTxt("PyramidalFlowFilter.width(): expecting 2 input parameters.");

        plhs[0] = mxCreateDoubleScalar(instance->width());
        return;
    }

    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized: ");
}
