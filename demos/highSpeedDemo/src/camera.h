/**
 * \file camera.h
 * \brief Camera interface.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#include <string>
#include <memory>

#include <QObject>

#include <pylon/PylonIncludes.h>

#include <flowfilter/image.h>


Q_DECLARE_METATYPE(flowfilter::image_t);

class CameraWrapper : public QObject {

    Q_OBJECT

private:
    std::string __camConfigFilePath;
    std::unique_ptr<Pylon::CInstantCamera> __camera;
    
public:
    /**
     * \brief creates a camera object using configuration parameters
     *    provided at configFilePath
     *
     */
    CameraWrapper(const std::string& configFilePath);
    ~CameraWrapper();

public:
    /**
     * \brief open and configures the camera according to the configuration file.
     */
    void open();
    void close();
    void startGrabbing();
    void stopGrabbing();

    void setFrameRate(float framerate);

    void registerImageHandler(Pylon::CImageEventHandler* handler);

    int width();
    int height();

    void printInfo();

public slots:
    void trigger();
};


/**
 * \brief Image handler to interface between Pylon events and the rest of the application
 */
class ImageHandler :    public QObject,
                        public Pylon::CImageEventHandler {

    Q_OBJECT

private:
    QByteArray __imageData;
    bool __firstGrab;
    flowfilter::image_t __image;
    

public:
    ImageHandler();
    ~ImageHandler();

public:
    void OnImageGrabbed(Pylon::CInstantCamera& camera, const Pylon::CGrabResultPtr& grabResult) override;

    //#################################
    // SIGNALS
    //#################################
signals:
    // void imageReceived(const QByteArray& data, const int height, const int width, const int pitch);
    void imageReceived(flowfilter::image_t image);
};


#endif // CAMERA_H_