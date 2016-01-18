# Optical-flow-filter

A real time optical flow algorithm implemented on GPU.


# Build and Installation

    git clone https://github.com/jadarve/optical-flow-filter.git
    cd optical-flow-filter
    mkdir build
    cd build
    cmake ..
    make
    sudo make install 
    
The library and header files will be installed at **/usr/local/lib** and **/usr/local/include** respectively.

##Python Wrappers

A python package with wrappers to the C++ library is available at **optical-flow-filter/python/** folder. The wrappers have been developed and build using Cython 0.23.4.

    cd optical-flow-filter/python/
    python setup.py build
    sudo python setup.py install

See **notebooks/** folder for usage examples.

# Demo Applications

## flowWebCam

This demo computes optical flow from a webcam. It uses OpenCV to access the camera video stream and to display the computed flow. The instructions to build the demo are the following:

    cd optical-flow-filter/demos/flowWebCam
    mkdir build
    cd build
    cmake ..
    make
    ./flowWebCam


## highSpeedDemo

This demo interfaces a Basler camera, in our case an acA2000-165um, with the GPU optical flow algorithm, and displays the color encoded flow.

    cd optical-flow-filter/demos/highSpeedDemo
    mkdir build
    cd build
    cmake ..
    make

To run the application, it is necessary to specify the camera properties file, as follows

    ./highSpeedDemo -c ../acA2000-165um_binSkip.pfs

Other optional arguments are:
    
    ./highSpeedDemo -h
    
    -h, --help             Displays this help.
    -v, --version          Displays version information.
    -c, --config <file>    Camera configuration file.
    -l, --levels <int>     Flow filter pyramid levels (default 2).
    -r, --rate <int>       Camera frame rate (default 300).
    -m, --maxflow <float>  Maximum optical flow (default 4.0).


# References

 Article under review on Robotics and Automation Letters (RA-L)

