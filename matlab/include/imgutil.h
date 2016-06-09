#ifndef IMGUTIL_H_
#define IMGUTIL_H_

#include "mex.h"

#include "flowfilter/image.h"
#include "flowfilter/gpu/image.h"


void printImageWrappedInfo(const char* preamble, const flowfilter::image_t& img);
void printGPUImageInfo(const char* preamble, const flowfilter::gpu::GPUImage& img);


mxArray* createMxArray(const flowfilter::gpu::GPUImage& img);
mxArray* createMxArray(const int height, const int width, mxClassID arrClass);
mxArray* createMxArray(const int height, const int width, const int depth, mxClassID arrClass);


flowfilter::image_t wrapMxImage(const mxArray* img);


#endif // IMGUTIL_H_
