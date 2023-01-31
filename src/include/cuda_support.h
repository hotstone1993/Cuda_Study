#include <cuda_runtime.h>
#include <device_launch_parameters.h>

inline void initCUDA() {
    int deviceCount = 0;
    if (cudaSuccess != cudaGetDeviceCount(&deviceCount))
    {
        throw "Do you have a CUDA-capable GPU installed?";
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        throw "cudaSetDevice failed!";
    }
}