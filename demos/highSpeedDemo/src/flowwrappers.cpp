/**
 * \file flowwrappers.h
 * \brief QT wrappers for flowfilter classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <vector>

#include "flowwrappers.h"

using namespace std;
using namespace flowfilter::gpu;

//#########################################################
// PyramidalFlowFilterWrapper
//#########################################################

PyramidalFlowFilterWrapper::PyramidalFlowFilterWrapper():
    QObject() {

}


PyramidalFlowFilterWrapper::PyramidalFlowFilterWrapper(const int height, const int width, const int levels):
    QObject() {

    __filter = std::make_shared<PyramidalFlowFilter>(height, width, levels);

    for(int h = 0; h < levels; h ++) {
        __filter->setGamma(h, 50.0f);
        __filter->setSmoothIterations(h, 4);
    }
}


PyramidalFlowFilterWrapper::~PyramidalFlowFilterWrapper() {
    // nothing to do
}


shared_ptr<PyramidalFlowFilter> PyramidalFlowFilterWrapper::getFilter() {
    return __filter;
}


void PyramidalFlowFilterWrapper::imageReceived(flowfilter::image_t image) {

    __filter->loadImage(image);
    __filter->compute();

    std::cout << "elapsed time: " << __filter->elapsedTime() << " ms\r" ;

    emit flowComputed();
}

