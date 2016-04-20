message(STATUS "entering src/gpu/device folder")

add_gpu_sources(
    imagemodel_k.cu
    propagation_k.cu
    update_k.cu
    flowsmoothing_k.cu
    pyramid_k.cu
    display_k.cu
    misc_k.cu
    rotation_k.cu
)