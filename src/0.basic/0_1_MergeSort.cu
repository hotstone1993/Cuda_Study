#include "0_1_MergeSort.cuh"

template <class T1, class T2>
void basic::merge::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    // checkCudaError(cudaMemcpy(outputs[HOST_OUTPUT1], outputs[DEVICE_OUTPUT1], SIZE * SIZE * sizeof(T2), cudaMemcpyDeviceToHost), "cudaMemcpy failed! (Device to Host) - ");
}

template void basic::merge::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);