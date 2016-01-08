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

Demo applications are available under the **demos** folder. To build  **flowWebCam** application to compute optical flow from webCam video follow the instructions bellow. These application requires OpenCV to be installed in the system.

    cd optical-flow-filter/demos/flowWebCam
    mkdir build
    cd build
    cmake ..
    make
    ./flowWebCam
    

# References

 Article under review on Robotics and Automation Letters (RA-L)

