message(STATUS "entering src/gpu folder")

add_gpu_sources(
    # CORE MODULES
    error.cu
    image.cu
    util.cu
    pipeline.cu
    camera.cu

    # ALGORITHMS DEPENDING ON CORE MODULES
    imagemodel.cu
    propagation.cu
    update.cu
    flowsmoothing.cu
    flowfilter.cu
    pyramid.cu
    display.cu
    rotation.cu
)

# process CMakeLists.txt in device folder
add_subdirectory(device)

# propagate GPU_SRCS to top level
set (GPU_SRCS ${GPU_SRCS} PARENT_SCOPE)
