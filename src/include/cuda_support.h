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
    cudaError_t cudaStatus = cudaMalloc((void**)&ptr, size * sizeof(TYPE)); \
    if (cudaStatus != cudaSuccess) { \
        throw "cudaMalloc failed!"; \
    }\
}

#define CUDA_ALLOC(INPUT_SIZE, INPUT_TYPE, OUTPUT_SIZE, OUTPUT_TYPE) { \
    host_input = new INPUT_TYPE[INPUT_SIZE]; \
    host_output = new OUTPUT_TYPE[OUTPUT_SIZE]; \
    CUDA_MALLOC(device_input, INPUT_SIZE, INPUT_TYPE); \
    CUDA_MALLOC(device_output, OUTPUT_SIZE, OUTPUT_TYPE); \
}

#define CUDA_DEALLOC() { \
    delete[] host_input; \
    delete[] host_output; \
    cudaFree(device_input); \
    cudaFree(device_output); \
}