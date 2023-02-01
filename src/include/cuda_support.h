#include <cuda_runtime.h>
#include <device_launch_parameters.h>

inline void initCUDA() {
    int deviceCount = 0;
    if (cudaSuccess != cudaGetDeviceCount(&deviceCount))
    {
        throw "Do you have a CUDA-capable GPU installed?";
    }
    // only use 
    if (cudaSetDevice(0) != cudaSuccess) {
        throw "cudaSetDevice failed!";
    }
}

#define CUDA_MALLOC(ptr, size, TYPE) {\
    cudaMalloc((void**)&ptr, size * sizeof(TYPE)); \
}
