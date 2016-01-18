/**
 * \file main.cpp
 * \brief Application entry point.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <vector>
#include <memory>

#include <pylon/PylonIncludes.h>

#include <QApplication>
#include <QThread>
#include <QTimer>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QDebug>

#include <flowfilter/image.h>
#include <flowfilter/gpu/flowfilter.h>

#include "camera.h"
#include "flowwidgets.h"
#include "flowwrappers.h"


using namespace std;
using namespace flowfilter::gpu;


int main(int argc, char **argv)
{
    // initializes Pylon
    Pylon::PylonAutoInitTerm autoInitTerm;

    // QT Application
    QApplication app (argc, argv);
    app.setApplicationName("highSpeedDemo");
    app.setApplicationVersion("1.0");

    // register image_t as QT MetaType. This enables to use
    // signals and slots with image_t arguments
    qRegisterMetaType<flowfilter::image_t>("image_t");
    int image_t_ID = qRegisterMetaType<flowfilter::image_t>();
    // std::cout << "image_t registered: " << QMetaType::isRegistered(image_t_ID) << std::endl;

    // Command line argument parsing
    QCommandLineParser parser;
    parser.setApplicationDescription("High speed optical flow demo");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addOption(QCommandLineOption{{"c", "config"}, "Camera configuration file.", "file", ""});
    parser.addOption(QCommandLineOption{{"l", "levels"}, "Flow filter pyramid levels (default 2).", "int", "2"});
    parser.addOption(QCommandLineOption{{"r", "rate"}, "Camera frame rate (default 300).", "int", "300"});
    parser.addOption(QCommandLineOption{{"m", "maxflow"}, "Maximum optical flow (default 4.0).", "float", "4.0"});
    
    // process command line arguments
    parser.process(app);
    std::cout << "QT Version: " << QT_VERSION_STR << std::endl;


    //#################################
    // ARGUMENTS PROCESSING
    //#################################
    bool framerateOK = false;
    int framerate = parser.value("rate").toInt(&framerateOK);
    if(!framerateOK) {
        std::cerr << "ERROR: error parsing input frame rate: " << parser.value("rate").toStdString() << std::endl;
        exit(-1);
    }

    bool levelsOK = false;
    int levels = parser.value("levels").toInt(&levelsOK);
    if(!levelsOK) {
        std::cout << "ERROR: error parsing pyramid levels argument, using default " << std::endl;
        exit(-1);
    }

    bool maxflowOK = false;
    float maxflow = parser.value("maxflow").toFloat(&maxflowOK);
    if(!maxflowOK) {
        std::cerr << "ERROR: error parsing maxflow argument: " << parser.value("maxflow").toStdString() << std::endl;
        exit(-1);
    }

    std::cout << "Camera configuration file: " << parser.value("config").toStdString() << std::endl;
    std::cout << "Frame rate: " << framerate << " Hz." << std::endl;
    std::cout << "Pyramid levels: " << levels << std::endl;
    std::cout << "Maximum optical flow: " << maxflow << std::endl;


    //#################################
    // THREADS
    //#################################
    QThread cameraGrabThread;       // camera grab thread
    QThread gpuWorkerThread;        // GPU filter thread
    QThread guiThread;

    cameraGrabThread.start(QThread::TimeCriticalPriority);
    gpuWorkerThread.start(QThread::TimeCriticalPriority);
    guiThread.start(QThread::TimeCriticalPriority);


    //#################################
    // GUI REFRESH TIMERS
    //#################################
    QTimer screenRefreshTimer;
    

    //#################################
    // CAMERA
    //#################################

    CameraWrapper camera(parser.value("config").toStdString());
    camera.open();
    camera.startGrabbing();
    camera.setFrameRate((float)framerate);
    camera.moveToThread(&cameraGrabThread);
    camera.printInfo();

    //#################################
    // IMAGE HANDLERS
    //#################################
    // Handler to feed the gpuFilter
    ImageHandler gpuImageHandler;
    gpuImageHandler.moveToThread(&gpuWorkerThread);
    camera.registerImageHandler(&gpuImageHandler);

    // Handler for user interface
    ImageHandler guiImageHandler;
    guiImageHandler.moveToThread(&guiThread);
    camera.registerImageHandler(&guiImageHandler);
    

    //#################################
    // OPTICAL FLOW FILTER
    //#################################
    PyramidalFlowFilterWrapper filter(camera.height(), camera.width(), levels);
    filter.moveToThread(&gpuWorkerThread);
    QObject::connect(&gpuImageHandler, &ImageHandler::imageReceived, &filter, &PyramidalFlowFilterWrapper::imageReceived);

    shared_ptr<PyramidalFlowFilter> f = filter.getFilter();
    f->setMaxFlow(maxflow);

    //#################################
    // WIDGETS
    //#################################
    CameraWidget camWidget;
    QObject::connect(&guiImageHandler, &ImageHandler::imageReceived, &camWidget, &CameraWidget::imageReceived);
    QObject::connect(&screenRefreshTimer, &QTimer::timeout, &camWidget, &CameraWidget::refresh);
    camWidget.show();


    FlowToColorWidget flowToColor;
    flowToColor.connectTo(&filter);
    QObject::connect(&filter, &PyramidalFlowFilterWrapper::flowComputed, &flowToColor, &FlowToColorWidget::flowComputed);
    QObject::connect(&screenRefreshTimer, &QTimer::timeout, &flowToColor, &FlowToColorWidget::refresh);
    flowToColor.show();
    flowToColor.setMaxFlow(maxflow);

    // start refresh of GUI widgets
    screenRefreshTimer.start(20);

    return app.exec();
}