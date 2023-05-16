#include "0_2_LinearSearch.cuh"


__global__ void linearSearch(TARGET_INPUT_TYPE* input, TARGET_INPUT_TYPE target, TARGET_OUTPUT_TYPE* result) {
    TARGET_OUTPUT_TYPE idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (TARGET_OUTPUT_TYPE i = idx; i < SIZE; i += gridDim.x * blockDim.x) {
        if (input[i] == target) {
            atomicMin(result, i);
            break;
        }
    }
}

template <class T1, class T2>
void basic::linear_search::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    int stride = 32;
    dim3 gridDim(SIZE - (stride * THREADS - 1) / (stride * THREADS));
    dim3 blockDim(THREADS);
    linearSearch<<<gridDim, blockDim>>>(inputs[DEVICE_INPUT], inputs[TARGET_NUMBER][0], outputs[DEVICE_OUTPUT]);
    cudaDeviceSynchronize();

    checkCudaError(cudaGetLastError(), "Linear Search launch failed - ");

    checkCudaError(cudaMemcpy(outputs[HOST_OUTPUT], outputs[DEVICE_OUTPUT], sizeof(T2), cudaMemcpyDeviceToHost), "cudaMemcpy failed! (Device to Host) - ");
}

template void basic::linear_search::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);