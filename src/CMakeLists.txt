message(STATUS "entering src folder")

add_gpu_sources (
    image.cpp
    colorwheel.cpp
)

# process CMakeLists.txt in gpu folder
add_subdirectory(gpu)

# propagate GPU_SRCS to top level
set (GPU_SRCS ${GPU_SRCS} PARENT_SCOPE)
