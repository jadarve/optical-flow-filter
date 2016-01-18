cmake_minimum_required(VERSION 2.8.11)
project(highSpeedDemo)


#################################################
# MACROS
#################################################
macro (add_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND SRCS "${_relPath}/${_src}")
        else()
            list (APPEND SRCS "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (SRCS ${SRCS} PARENT_SCOPE)
    endif()
endmacro()


#################################################
# COMPILER SETTINGS
#################################################
set (CMAKE_CXX_COMPILER         "g++")
set (CMAKE_CXX_FLAGS            "-std=c++11 -O2 -Wall")


#################################################
# REQUIREMENTS
#################################################

# Find the QtWidgets library
find_package(Qt5Widgets REQUIRED)
find_package(CUDA REQUIRED)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(AUTOMOC_MOC_OPTIONS "-I${CMAKE_SOURCE_DIR}/include")
# set(CMAKE_INCLUDE_CURRENT_DIR ON)


# Pylon4
set(PYLON_ROOT /opt/pylon4)

set(PYLON_INCLUDE
    ${PYLON_ROOT}/include
    ${PYLON_ROOT}/genicam/library/CPP/include
)

link_directories(
    ${PYLON_ROOT}/lib64
    ${PYLON_ROOT}/genicam/bin/Linux64_x64
    ${PYLON_ROOT}/genicam/bin/Linux64_x64/GenApi/Generic
)

set(PYLON_LIBS
    pylonbase
    pylonutility
    GenApi_gcc40_v2_3
    GCBase_gcc40_v2_3
    Log_gcc40_v2_3
    MathParser_gcc40_v2_3
    Xerces-C_gcc40_v2_7_1
    log4cpp_gcc40_v2_3
    XMLLoader_gcc40_v2_3
)



#################################################
# SOURCE FILES
#################################################
add_subdirectory(src)
# add_subdirectory(include)

# ${CMAKE_SOURCE_DIR}/include
include_directories(
    ${PYLON_INCLUDE}
)

# qt5_wrap_cpp(moc_camera.cpp include/camera.h)
# qt5_generate_moc(include/camera.h moc_camera.cpp)
#################################################
# TARGETS
#################################################

# Tell CMake to create the helloworld executable
add_executable(highSpeedDemo ${SRCS})

# Use the Widgets module from Qt 5.
target_link_libraries(highSpeedDemo
    ${PYLON_LIBS}
    flowfilter_gpu
    Qt5::Widgets
)

qt5_use_modules(highSpeedDemo Widgets)