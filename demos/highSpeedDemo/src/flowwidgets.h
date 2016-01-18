/**
 * \file flowwidgets.h
 * \brief Visualization widgets
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef FLOWWIDGETS_H_
#define FLOWWIDGETS_H_

#include <memory>

#include <QWidget>
#include <QImage>

#include <pylon/PylonIncludes.h>

#include <flowfilter/image.h>
#include <flowfilter/gpu/display.h>

#include "flowwrappers.h"



class CameraWidget : public QWidget {

    Q_OBJECT

private:
    /** Contains the bytes of the image to be displayed */
    QImage __imageBuffer;
    bool __firstImageGrabbed;

public:
    CameraWidget(QWidget* parent = 0);
    ~CameraWidget();

public slots:
    void imageReceived(flowfilter::image_t image);

    /**
     * \brief slot to refresh the screen content.
     */
    void refresh();

protected:
    void paintEvent(QPaintEvent* event) Q_DECL_OVERRIDE;
};



class FlowToColorWidget : public QWidget {

    Q_OBJECT


public:
    FlowToColorWidget(QWidget* parent = 0);
    ~FlowToColorWidget();


public:
    void connectTo(PyramidalFlowFilterWrapper* filter);
    void setMaxFlow(const float maxflow);


public slots:
    /**
     * \brief Slot to receive events on new flow field avaiable.
     */
    void flowComputed();

    /**
     * \brief slot to refresh the screen content.
     */
    void refresh();

protected:
    void paintEvent(QPaintEvent* event) Q_DECL_OVERRIDE;

private:
    /** Tells if __flowToColor is connected */
    bool __connected;

    bool __firstCompute;
    flowfilter::gpu::FlowToColor __flowToColor;

    QImage __flowColorImage;
    flowfilter::image_t __flowColorImageWrapped;

};

#endif /* FLOWWIDGETS_H_ */
