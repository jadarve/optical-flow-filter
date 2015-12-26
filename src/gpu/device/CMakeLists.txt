message(STATUS "entering src/gpu/device folder")

add_gpu_sources(
    imagemodel_k.cu
    propagation_k.cu
    update_k.cu
    flowsmoothing_k.cu
)