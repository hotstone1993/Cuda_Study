#ifndef UTILS
#define UTILS

#include <concepts>
#include <cmath>
#include <limits>
#include <iostream>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include "ThreadPool.h"
#include "EventTimer.h"

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

inline void checkCudaError(cudaError_t status, const char* prefix) {
    if (status != cudaSuccess) {
        std::string message = prefix;
        message += cudaGetErrorString(status);
        throw std::runtime_error(message);
    }
}

#define CUDA_MALLOC(ptr, size, TYPE) {\
    cudaError_t cudaStatus = cudaMalloc((void**)&ptr, size * sizeof(TYPE)); \
    if (cudaStatus != cudaSuccess) { \
        throw std::runtime_error("cudaMalloc failed!"); \
    }\
}

template<class T>
inline T divideUp(T totalSize, T threadCount) {
    return totalSize % threadCount != 0 ? totalSize / threadCount + 1 : totalSize / threadCount;
}

// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp) {
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
        // unless the result is subnormal
        || std::fabs(x-y) < std::numeric_limits<T>::min();
}

__device__ uint32_t __mysmid();
__device__ uint32_t __mywarpid();
__device__ uint32_t __mylaneid();

#endif // UTILS