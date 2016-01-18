/**
 * \file camera.cpp
 * \brief Camera interface.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <exception>
#include <iostream>
#include <cstring>

#include "camera.h"


using namespace std;
using namespace Pylon;
using namespace GenApi;


//#########################################################
// CameraWrapper
//#########################################################

CameraWrapper::CameraWrapper(const std::string& configFilePath):
    QObject() {

    // configuration file
    __camConfigFilePath = configFilePath;

    // use the first camera connected to the computer.
    __camera = unique_ptr<Pylon::CInstantCamera>(new Pylon::CInstantCamera(
                   Pylon::CTlFactory::GetInstance().CreateFirstDevice()));

    __camera->RegisterConfiguration(new Pylon::CSoftwareTriggerConfiguration,
                                    Pylon::RegistrationMode_ReplaceAll, Pylon::Cleanup_Delete);

    __camera->MaxNumBuffer = 10;
}

CameraWrapper::~CameraWrapper() {

    // nothing to do
}

void CameraWrapper::open() {

    cout << "Camera::open()" << endl;

    try {
        __camera->Open();
        CFeaturePersistence::Load(__camConfigFilePath.c_str(), &__camera->GetNodeMap(), true);



    } catch (GenICam::GenericException& e) {
        cerr << "error opening camera and loading configuration file: " << __camConfigFilePath << endl;
        cerr << "message: " << e.GetDescription() << endl;
    }
}

void CameraWrapper::printInfo() {

    try {
        cout << "width:\t\t\t\t" << CIntegerPtr(__camera->GetNodeMap().GetNode("Width"))->GetValue() << endl;
        cout << "height:\t\t\t\t" << CIntegerPtr(__camera->GetNodeMap().GetNode("Height"))->GetValue() << endl;
        cout << "offset X:\t\t\t" << CIntegerPtr(__camera->GetNodeMap().GetNode("OffsetX"))->GetValue() << endl;
        cout << "offset Y:\t\t\t" << CIntegerPtr(__camera->GetNodeMap().GetNode("OffsetY"))->GetValue() << endl;
        cout << "resulting frame rate:\t\t" << CFloatPtr(__camera->GetNodeMap().GetNode("ResultingFrameRate"))->GetValue() << endl;

        // configure the device link limit mode as off
        CEnumerationPtr(__camera->GetNodeMap().GetNode("DeviceLinkThroughputLimitMode"))->SetIntValue(0);

        // enable software trigger mode
//      CEnumerationPtr(__camera->GetNodeMap().GetNode("TriggerMode"))->SetIntValue(1);

        cout << "link selector:\t\t\t" << CIntegerPtr(__camera->GetNodeMap().GetNode("DeviceLinkSelector"))->GetValue() << endl;
        cout << "link throughput limit mode:\t" << CEnumerationPtr(__camera->GetNodeMap().GetNode("DeviceLinkThroughputLimitMode"))->GetIntValue() << endl;
        cout << "link throughput limit:\t\t" << CIntegerPtr(__camera->GetNodeMap().GetNode("DeviceLinkThroughputLimit"))->GetValue() << endl;
        cout << "link current throughput:\t" << CIntegerPtr(__camera->GetNodeMap().GetNode("DeviceLinkCurrentThroughput"))->GetValue() << endl;

    } catch (GenICam::GenericException& e) {
        cerr << "ERROR: CameraWrapper::printInfo(): error opening camera and loading configuration file: " << __camConfigFilePath << endl;
        cerr << "message: " << e.GetDescription() << endl;
    }
}

void CameraWrapper::close() {
    __camera->Close();
}

void CameraWrapper::startGrabbing() {
//  __camera->RegisterConfiguration(new Pylon::CSoftwareTriggerConfiguration, Pylon::RegistrationMode_ReplaceAll, Pylon::Cleanup_Delete);
    __camera->StartGrabbing(Pylon::GrabStrategy_OneByOne, Pylon::GrabLoop_ProvidedByInstantCamera);
}

void CameraWrapper::stopGrabbing() {
    __camera->StopGrabbing();
}

void CameraWrapper::setFrameRate(float framerate) {
    CBooleanPtr(__camera->GetNodeMap().GetNode("AcquisitionFrameRateEnable"))->SetValue(true);
    CFloatPtr(__camera->GetNodeMap().GetNode("AcquisitionFrameRate"))->SetValue(framerate);
}

void CameraWrapper::trigger() {
    try {
        __camera->WaitForFrameTriggerReady(10, Pylon::TimeoutHandling_ThrowException);
        __camera->ExecuteSoftwareTrigger();
    } catch (GenICam::GenericException& e) {
        cerr << "CameraWrapper::triggerCamera(): error: " << e.GetDescription() << endl;
    }
}

void CameraWrapper::registerImageHandler(Pylon::CImageEventHandler* handler) {

    if (handler != NULL) {
        __camera->RegisterImageEventHandler(handler, Pylon::RegistrationMode_Append, Pylon::Cleanup_None);
    }
}

int CameraWrapper::width() {
    return CIntegerPtr(__camera->GetNodeMap().GetNode("Width"))->GetValue();
}

int CameraWrapper::height() {
    return CIntegerPtr(__camera->GetNodeMap().GetNode("Height"))->GetValue();
}


//#########################################################
// ImageHandler
//#########################################################

ImageHandler::ImageHandler() {
    __firstGrab = true;
}

ImageHandler::~ImageHandler() {
    // nothing to do
}

void ImageHandler::OnImageGrabbed(Pylon::CInstantCamera& camera, const Pylon::CGrabResultPtr& grabResult) {

    if (__firstGrab) {

        if (grabResult->GrabSucceeded()) {
            __imageData = QByteArray(grabResult->GetImageSize(), 0);

            // wrap grabResult and allocated __imageData in __image
            __image.height = grabResult->GetHeight();
            __image.width = grabResult->GetWidth();
            grabResult->GetStride(__image.pitch);
            __image.depth = 1;
            __image.itemSize = 1;
            __image.data = (void*)__imageData.data();


            // the image has been configured successfully
            __firstGrab = false;
        }
    }

    // emit an imageReceived event to all subscribers of this handler
    if (grabResult->GrabSucceeded()) {
        memcpy((void*)__imageData.data(), grabResult->GetBuffer(), grabResult->GetImageSize());

        // emit signal notifying a new image has been grabbed
        // emit imageReceived(__imageData, __height, __width, __imagePitch);
        emit imageReceived(__image);
    }
}