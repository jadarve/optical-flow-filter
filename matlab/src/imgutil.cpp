
#include "imgutil.h"


#include <vector>


using namespace std;
using namespace flowfilter;
using namespace flowfilter::gpu;


void printImageWrappedInfo(const char* preamble, const image_t& img) {
    mexPrintf("%s: array shape: [%d, %d, %d] item size: %d\n", preamble,
        img.height, img.width, img.depth, img.itemSize);
}


void printGPUImageInfo(const char* preamble, const GPUImage& img) {
    mexPrintf("%s: array shape: [%d, %d, %d] item size: %d\n", preamble,
        img.height(), img.width(), img.depth(), img.itemSize());   
}


mxArray* createMxArray(const GPUImage& img) {

    int ndim = img.depth() > 1? 3 : 2;
    vector<int> dims(ndim);

    // this order for [height, width] is used to fill the returned Matlab array as the
    // transpose of the GPU image. This transforms GPU buffer in row-major order to
    // Matlab column-major order
    dims[0] = img.width();
    dims[1] = img.height();

    if(ndim > 2) {
        dims[2] = img.depth();
    }

    mxClassID arrClass = img.itemSize() == 1? mxUINT8_CLASS : mxSINGLE_CLASS;

    return mxCreateNumericArray(ndim, &dims[0], arrClass, mxREAL);
}


mxArray* createMxArray(const int height, const int width, const int depth, mxClassID arrClass) {

    // this order for [height, width] is used to fill the returned Matlab array as the
    // transpose of the GPU image. This transforms GPU buffer in row-major order to
    // Matlab column-major order
    vector<int> dims {width, height, depth};
    
    return mxCreateNumericArray(3, &dims[0], arrClass, mxREAL);
}


mxArray* createMxArray(const int height, const int width, mxClassID arrClass) {

    // this order for [height, width] is used to fill the returned Matlab array as the
    // transpose of the GPU image. This transforms GPU buffer in row-major order to
    // Matlab column-major order
    return mxCreateNumericMatrix(width, height, arrClass, mxREAL);
}


image_t wrapMxImage(const mxArray* img) {

    size_t ndim = mxGetNumberOfDimensions(img);
    if(ndim != 2 && ndim != 3) mexErrMsgTxt("wrapMxImage(): array parameter should have 2 or 3 dimensions.");

    const int* dims = mxGetDimensions(img);

    // wrap image into image_t type
    image_t img_w;

    // this order to read [height, width] assumes that the image is passed as the transpose
    // of the original parameter. This makes that a Matlab matrix in column-major order can
    // be read in row-major order by the flowfilter library and image values are properly
    // filled in GPU memory (row-major)
    img_w.height = dims[1];
    img_w.width = dims[0];
    img_w.depth = ndim == 2? 1 : dims[2];
    img_w.itemSize = mxGetElementSize(img);
    img_w.pitch = img_w.width*img_w.depth*img_w.itemSize;
    img_w.data = mxGetData(img);

    // printImageWrappedInfo("wrapMxImage():", img_w);

    return img_w;
}
