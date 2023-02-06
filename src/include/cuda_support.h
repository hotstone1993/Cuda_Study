#include <cuda_runtime.h>
#include <stdexcept>
#include <device_launch_parameters.h>

inline void initCUDA() {
    int deviceCount = 0;
    if (cudaSuccess != cudaGetDeviceCount(&deviceCount))
    {
        throw std::runtime_error("Do you have a CUDA-capable GPU installed?");
    }
    // only use 
    if (cudaSetDevice(0) != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice failed!");
    }
}


#define CUDA_MALLOC(ptr, size, TYPE) {\
    cudaError_t cudaStatus = cudaMalloc((void**)&ptr, size * sizeof(TYPE)); \
    if (cudaStatus != cudaSuccess) { \
        throw std::runtime_error("cudaMalloc failed!"); \
    }\
}