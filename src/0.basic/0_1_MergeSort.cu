#include "0_1_MergeSort.cuh"

__global__ void mergeSort(TARGET_INPUT_TYPE* input)
{
}

template <class T1, class T2>
void basic::merge::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    dim3 gridDim(SIZE - (THREADS - 1) / THREADS);
    dim3 blockDim(THREADS);

    mergeSort<<<gridDim, blockDim>>>(inputs[DEVICE_INPUT]);

    checkCudaError(cudaMemcpy(inputs[HOST_INPUT], inputs[DEVICE_INPUT], SIZE * sizeof(T2), cudaMemcpyDeviceToHost), "cudaMemcpy failed! (Device to Host) - ");
}

template void basic::merge::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);