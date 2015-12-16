cmake_minimum_required(VERSION 2.8.11)
project(optical-flow-filter)


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
set (CMAKE_CXX_COMPILER         "g++")
set (CMAKE_CXX_FLAGS            "-std=c++11 -flto -O2 -Wall")


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

    # NVCC compiler flags
    set(CUDA_NVCC_FLAGS             "-std=c++11")
    set(CUDA_PROPAGATE_HOST_FLAGS ON)    

endif(CUDA_FOUND)


#################################################
# TARGETS
#################################################

if(CUDA_FOUND)
    
    # flowfilter_gpu library with CUDA implementation
    cuda_add_library(flowfilter_gpu SHARED ${GPU_SRCS})

endif(CUDA_FOUND)
