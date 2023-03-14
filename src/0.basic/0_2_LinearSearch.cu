#include "0_2_LinearSearch.cuh"

namespace cg = cooperative_groups;

__global__ void linearSearch() {
    
}

template <class T1, class T2>
void basic::linear_search::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    dim3 gridDim(SIZE - (THREADS - 1) / THREADS);
    dim3 blockDim(THREADS);

    checkCudaError(cudaMemcpy(inputs[HOST_INPUT], outputs[DEVICE_OUTPUT], SIZE * sizeof(T2), cudaMemcpyDeviceToHost), "cudaMemcpy failed! (Device to Host) - ");
}

template void basic::linear_search::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);