#include "0_1_MergeSort.cuh"

namespace cg = cooperative_groups;

__device__ void sort(TARGET_INPUT_TYPE* input, TARGET_INPUT_TYPE* temp, unsigned int start, unsigned int idx, unsigned int stride, unsigned int size) {
    unsigned int left = start;
    unsigned int right = start + (stride / 2);
    const unsigned int mid = right < SIZE ? right : SIZE;
    const unsigned int end = start + stride < SIZE ? start + stride : SIZE;

    while (left < mid || right < end) {
        if (left < mid && right < end) {
            if (input[left] < input[right]) {
                temp[idx++] = input[left++];
            } else {
                temp[idx++] = input[right++];
            }
        } else if (left >= mid && right < end) {
            temp[idx++] = input[right++];
        } else if (right >= end && left < mid) {
            temp[idx++] = input[left++];
        }
    }
}

__global__ void mergeSortStep1(TARGET_INPUT_TYPE* input)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ TARGET_INPUT_TYPE temp[THREADS];
    unsigned int stride = 2;

    while (stride < THREADS) {
        if (idx % stride == 0) {
            sort(input, temp, idx, threadIdx.x, stride, THREADS);
        }
        __syncthreads();
        if (idx < SIZE) {
            input[idx] = temp[threadIdx.x];
        }
        __syncthreads();
        stride *= 2;
    }
}

__global__ void mergeSortStep2(TARGET_INPUT_TYPE* input, TARGET_OUTPUT_TYPE* output, unsigned int stride)
{
    bool flag = true;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    TARGET_INPUT_TYPE* buffer1 = input;
    TARGET_INPUT_TYPE* buffer2 = output;

    while (stride / 2 < SIZE) {
        if (idx % stride == 0) {
            if (flag) {
                sort(buffer1, buffer2, idx, idx, stride, SIZE);
            } else {
                sort(buffer2, buffer1, idx, idx, stride, SIZE);
            }
        }
        __syncthreads();
        flag = !flag;
        stride *= 2;
    }
    
    if (flag && idx < SIZE) {
        output[idx] = input[idx];
    }
}

template <class T1, class T2>
void basic::merge::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    dim3 gridDim(SIZE - (THREADS - 1) / THREADS);
    dim3 blockDim(THREADS);

    mergeSortStep1<<<gridDim, blockDim>>>(inputs[DEVICE_INPUT]);
    mergeSortStep2<<<gridDim, blockDim>>>(inputs[DEVICE_INPUT], outputs[DEVICE_OUTPUT], THREADS);
    cudaDeviceSynchronize();

    checkCudaError(cudaGetLastError(), "Merge Sort launch failed - ");

    checkCudaError(cudaMemcpy(inputs[HOST_INPUT], outputs[DEVICE_OUTPUT], SIZE * sizeof(T2), cudaMemcpyDeviceToHost), "cudaMemcpy failed! (Device to Host) - ");
}

template void basic::merge::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);