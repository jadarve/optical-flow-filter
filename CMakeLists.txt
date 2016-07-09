cmake_minimum_required(VERSION 2.8.11)
project(optical-flow-filter)


#################################################
# BUILDING ON WINDOWS
# 
# For X64
# 
# mkdir build64 & cd build64
# cmake -G "Visual Studio 12 2013 Win64" ..
# cmake --build . --config Release
# 
# 
# For x86
# 
# mkdir build & cd build
# cmake -G "Visual Studio 12 2013"
# cmake --build . --config Release
#################################################


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


macro (add_gpu_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        # message(STATUS "source: " ${_src})
        if (_relPath)
            list (APPEND GPU_SRCS "${_relPath}/${_src}")
        else()
            list (APPEND GPU_SRCS "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (GPU_SRCS ${GPU_SRCS} PARENT_SCOPE)
    endif()
    # message(STATUS "gpu sources: " ${GPU_SRCS})
endmacro()


#################################################
# LIBRARY SETTINGS
#################################################
set (LIBRARY_OUTPUT_PATH   "lib")

#################################################
# INCLUDE SETTINGS
#################################################
include_directories("${PROJECT_SOURCE_DIR}/include")


#################################################
# COMPILER SETTINGS
#################################################

if(WIN32)

    message(STATUS "Configuring for Windows system")
    
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT -DFLOWFILTERLIBRARY_EXPORTS")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd -DFLOWFILTERLIBRARY_EXPORTS")
    set(CMAKE_CXX_FLAGS            "-DFLOWFILTERLIBRARY_EXPORTS")

elseif(UNIX)

    message(STATUS "Configuring for Unix system")
    set (CMAKE_CXX_COMPILER         "g++")
    set (CMAKE_CXX_FLAGS            "-std=c++11 -flto -O3 -Wall")

endif(WIN32)




#################################################
# SOURCE FILES SETTINGS
#################################################
# process the CMakeLists.txt file at src/ folder
# it should add all the source files in the folder
add_subdirectory(src)


#################################################
# CUDA SETTINGS
#################################################
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    
    message(STATUS "found CUDA")
    message(STATUS "GPU sources: ${GPU_SRCS}")
    message(STATUS "HEADER FILES: ${HEADER_LIST}")


    if(WIN32)
        message(STATUS "Configuring CUDA for Windows")
        
        # THIS IS NECESSARY FOR WINDOWS BUILD
        set(CUDA_NVCC_FLAGS               "-DFLOWFILTERLIBRARY_EXPORTS")

        # flowfilter_gpu library with CUDA implementation
        cuda_add_library(flowfilter_gpu SHARED ${GPU_SRCS})

    elseif(UNIX)
        message(STATUS "Configuring CUDA for Unix")

        set(CUDA_NVCC_FLAGS             "-std=c++11 -O3")
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)

        cuda_add_library(flowfilter_gpu SHARED ${GPU_SRCS})

        # install
        install(
            TARGETS flowfilter_gpu
            LIBRARY DESTINATION lib
            COMPONENT library
        )

        # install header files
        install(
            DIRECTORY include/flowfilter
            DESTINATION include
        )

    endif(WIN32)

endif(CUDA_FOUND)
