message(STATUS "entering src/gpu folder")

add_gpu_sources(
    error.cu
    image.cu
    util.cu
    pipeline.cu
    imagemodel.cu
    propagation.cu
    update.cu
    flowsmoothing.cu
)

# process CMakeLists.txt in device folder
add_subdirectory(device)

# propagate GPU_SRCS to top level
set (GPU_SRCS ${GPU_SRCS} PARENT_SCOPE)
