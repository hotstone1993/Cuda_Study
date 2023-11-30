#include "3_0_Null_Stream.cuh"


__global__ void kernal1(TARGET_INPUT_TYPE* input, TARGET_OUTPUT_TYPE* output) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < gridDim.x * blockDim.x - 1) {
        output[idx] = (input[idx] + input[idx + 1]) / 2;
    } else {
        output[idx] = input[idx];
    }
}

template <class T1, class T2>
void basic::stream::run_comparison_target(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    dim3 gridDim(divideUp(SIZE, THREADS));
    dim3 blockDim(THREADS);

    kernal1<<<gridDim, blockDim>>>(inputs[DEVICE_INPUT1], inputs[DEVICE_OUTPUT1]);
    cudaDeviceSynchronize();
    
    checkCudaError(cudaGetLastError(), "Null Stream launch failed - ");

    checkCudaError(cudaMemcpy(outputs[HOST_OUTPUT1], outputs[DEVICE_OUTPUT1], sizeof(T2), cudaMemcpyDeviceToHost), "cudaMemcpy failed! (Device to Host) - ");
}

template void basic::stream::run_comparison_target(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);