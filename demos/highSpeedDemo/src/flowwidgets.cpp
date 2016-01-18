/**
 * \file flowwidgets.h
 * \brief Visualization widgets
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <cstring>
#include <iostream>
#include <chrono>


#include <QVector>
#include <QColor>
#include <QRgb>
#include <QPainter>

#include <flowfilter/gpu/flowfilter.h>

#include "flowwidgets.h"

using namespace std;
using namespace flowfilter::gpu;


//#########################################################
// CameraWidget
//#########################################################

CameraWidget::CameraWidget(QWidget* parent):
     QWidget(parent) {

    __firstImageGrabbed = true;

    cout << "CameraWidget::CameraWidget(): creating" << endl;
    cout << "camera widget completed" << endl;
}

CameraWidget::~CameraWidget() {
    // nothing to do
}


// void* CameraWidget::getImageBuffer() {
//     return (void*)__imageBuffer.bits();
// }


void CameraWidget::paintEvent(QPaintEvent* event) {

//  cout << "CameraWidget::paintEvent()" << endl;
    QPainter painter(this);
    painter.drawImage(QPoint(0,0), __imageBuffer);
}

void CameraWidget::imageReceived(flowfilter::image_t image) {

    if(__firstImageGrabbed) {
        __imageBuffer = QImage(image.width, image.height, QImage::Format_Indexed8);

        // create Gray scale color table for the display image
        QVector<QRgb> colormap(256);
        for(int i = 0; i < 256; i ++) {
            colormap[i] = QColor{i, i, i}.rgb();
        }
        __imageBuffer.setColorTable(colormap);

        // resize the widget to the image size
        resize(image.width, image.height);

        __firstImageGrabbed = false;
    }

    memcpy(__imageBuffer.bits(), image.data, image.pitch*image.height);
}

void CameraWidget::refresh() {
    update();
}


//#########################################################
// FlowToColorWidget
//#########################################################
FlowToColorWidget::FlowToColorWidget(QWidget* parent) :
    QWidget(parent) {

    __connected = false;
    __firstCompute = true;
}


FlowToColorWidget::~FlowToColorWidget() {
    // nothing to do
}


void FlowToColorWidget::connectTo(PyramidalFlowFilterWrapper* filter) {

    shared_ptr<PyramidalFlowFilter> f = filter->getFilter();

    __flowToColor = flowfilter::gpu::FlowToColor(f->getFlow(), 1.0f);

    // configure display buffer
    __flowColorImage = QImage(f->width(), f->height(), QImage::Format_RGBA8888);

    // wrapped __flowColorImage
    __flowColorImageWrapped.height = f->height();
    __flowColorImageWrapped.width = f->width();
    __flowColorImageWrapped.depth = 4;
    __flowColorImageWrapped.pitch = __flowColorImage.bytesPerLine();
    __flowColorImageWrapped.itemSize = 1;
    __flowColorImageWrapped.data = (void*)__flowColorImage.bits();

    // resize widget
    resize(f->width(), f->height());

    __connected = true;
}


void FlowToColorWidget::setMaxFlow(const float maxflow) {
    __flowToColor.setMaxFlow(maxflow);
}


void FlowToColorWidget::flowComputed() {

    if(__connected) {
        //  compute color code
        __flowToColor.compute();

        // download color to wrapped flowColorImage
        __flowToColor.downloadColorFlow(__flowColorImageWrapped);
    }
}


void FlowToColorWidget::refresh() {
    update();
}


void FlowToColorWidget::paintEvent(QPaintEvent* event) {

    QPainter painter(this);
    painter.drawImage(QPoint(0,0), __flowColorImage);
}