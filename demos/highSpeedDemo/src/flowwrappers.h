/**
 * \file flowwrappers.h
 * \brief QT wrappers for flowfilter classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWWRAPPERS_H_
#define FLOWWRAPPERS_H_

#include <memory>

#include <QObject>

#include <flowfilter/gpu/flowfilter.h>

class PyramidalFlowFilterWrapper : public QObject {

    Q_OBJECT

public:
    PyramidalFlowFilterWrapper();
    PyramidalFlowFilterWrapper(const int height, const int width, const int levels);
    ~PyramidalFlowFilterWrapper();


public:
    std::shared_ptr<flowfilter::gpu::PyramidalFlowFilter> getFilter();

signals:
    /** Signal triggered each time the filter performs computation of optical flow */
    void flowComputed();

public slots:
    void imageReceived(flowfilter::image_t image);


private:
    std::shared_ptr<flowfilter::gpu::PyramidalFlowFilter> __filter;

};


// class FlowToColorWrapper : QObject {

// };

#endif /* FLOWWRAPPERS_H_ */
