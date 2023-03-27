#include "0_1_MergeSort.cuh"

namespace cg = cooperative_groups;

__device__ void merge(TARGET_INPUT_TYPE* input, TARGET_INPUT_TYPE* temp, unsigned int start, unsigned int idx, unsigned int stride, unsigned int size) {
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

template <bool dir>
__device__ unsigned int getInclusivePositionByBinarySearch(TARGET_INPUT_TYPE* input, TARGET_INPUT_TYPE target, unsigned int bound, unsigned int stride) {
	unsigned int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = umin(pos + stride, bound);
		if ((dir && (input[newPos - 1] <= target)) || (!dir && (input[newPos - 1] > target))) {
			pos = newPos;
		}
	}
    return pos;
}

template <bool dir>
__device__ unsigned int getExclusivePositionByBinarySearch(TARGET_INPUT_TYPE* input, TARGET_INPUT_TYPE target, unsigned int bound, unsigned int stride) {
	unsigned int pos = 0;
	for (; stride > 0; stride >>= 1) {
		int newPos = umin(pos + stride, bound);
		if ((dir && (input[newPos - 1] < target)) || (!dir && (input[newPos - 1] > target))) {
			pos = newPos;
		}
	}
	return pos;
}

template <bool dir>
__global__ void mergeSortWithBinarySearch(TARGET_INPUT_TYPE* input) {
    __shared__ TARGET_INPUT_TYPE temp[2 * THREADS];
    unsigned int idx = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    temp[threadIdx.x] = input[idx];
    temp[blockDim.x + threadIdx.x] = input[blockDim.x + idx];

    for (unsigned int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
        unsigned int basePos = (threadIdx.x & (stride - 1));
        TARGET_INPUT_TYPE* base = temp + 2 * (threadIdx.x - basePos);
        __syncthreads();

        TARGET_INPUT_TYPE valueA = base[basePos];
        TARGET_INPUT_TYPE valueB = base[basePos + stride];
        unsigned int posA = getInclusivePositionByBinarySearch<dir>(base + stride, valueA, stride, stride);
        unsigned int posB = getExclusivePositionByBinarySearch<dir>(base, valueB, stride, stride);
        
        __syncthreads();
        base[posA] = valueA;
        base[posB] = valueB;
    }

    __syncthreads();
    input[idx] = temp[threadIdx.x];
    input[blockDim.x + idx] = temp[blockDim.x + threadIdx.x];
}

__global__ void mergeSortStep1(TARGET_INPUT_TYPE* input)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ TARGET_INPUT_TYPE temp[THREADS];
    unsigned int stride = 2;

    while (stride < THREADS) {
        if (idx % stride == 0) {
            merge(input, temp, idx, threadIdx.x, stride, THREADS);
        }
        __syncthreads();
        if (idx < SIZE) {
            input[idx] = temp[threadIdx.x];
        }
        __syncthreads();
        stride <<= 1;
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
                merge(buffer1, buffer2, idx, idx, stride, SIZE);
            } else {
                merge(buffer2, buffer1, idx, idx, stride, SIZE);
            }
        }
        __syncthreads();
        flag = !flag;
        stride <<= 1;
    }
    
    if (flag && idx < SIZE) {
        output[idx] = input[idx];
    }
}

template <bool dir>
__device__ inline bool compare(TARGET_INPUT_TYPE& a, TARGET_INPUT_TYPE& b) {
    if (dir) {
        return a > b;
    } else {
        return a <= b;
    }
}

__device__ inline bool compare(TARGET_INPUT_TYPE& a, TARGET_INPUT_TYPE& b, bool dir) {
    if (dir) {
        return a > b;
    } else {
        return a <= b;
    }
}

__device__ inline void swap(TARGET_INPUT_TYPE& a, TARGET_INPUT_TYPE& b) {
    TARGET_INPUT_TYPE temp = a;
    a = b;
    b = temp;
}

template <bool dir>
__device__ void evenSort(TARGET_INPUT_TYPE* input) {
    unsigned int idx = threadIdx.x;

    if (compare<dir>(input[2 * idx], input[2 * idx + 1])) {
        swap(input[2 * idx], input[2 * idx + 1]);
    }
}

template <bool dir>
__device__ void oddSort(TARGET_INPUT_TYPE* input) {
    unsigned int idx = threadIdx.x;

    if (idx == 0)
        return;

    if (compare<dir>(input[2 * idx - 1], input[2 * idx])) {
        swap(input[2 * idx - 1], input[2 * idx]);
    }
}

template <bool dir>
__global__ void blockLevelEvenOddSort(TARGET_INPUT_TYPE* input) {
    __shared__ TARGET_INPUT_TYPE sInput[2 * THREADS];

    sInput[threadIdx.x] = input[2 * blockDim.x * blockIdx.x + threadIdx.x];
    sInput[threadIdx.x + THREADS] = input[2 * blockDim.x * blockIdx.x + threadIdx.x + THREADS];
    __syncthreads();

    for (register unsigned i = 0; i < THREADS; ++i) {
        evenSort<dir>(sInput);
        __syncthreads();
        oddSort<dir>(sInput);
        __syncthreads();
    }

    input[2 * blockDim.x * blockIdx.x + threadIdx.x] = sInput[threadIdx.x];
    input[2 * blockDim.x * blockIdx.x + threadIdx.x + THREADS] = sInput[threadIdx.x + THREADS];
}

template <bool dir>
__global__ void bitonicSort(TARGET_INPUT_TYPE* input) {
    __shared__ TARGET_INPUT_TYPE sInput[2 * THREADS];

    sInput[threadIdx.x] = input[2 * blockDim.x * blockIdx.x + threadIdx.x];
    sInput[threadIdx.x + THREADS] = input[2 * blockDim.x * blockIdx.x + threadIdx.x + THREADS];

    for (size_t halfSize = 1; halfSize < blockDim.x; halfSize <<= 1) {
        bool curr = dir ^ ((threadIdx.x & halfSize) != 0);
        for (size_t stride = halfSize; stride > 0; stride >>= 1) {
            __syncthreads();
            size_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            if (compare(sInput[pos], sInput[pos + stride], dir)) {
                swap(sInput[pos], sInput[pos + stride]);
            }
        }
    }
	// bitonic merge step
	for (unsigned stride = blockDim.x; stride > 0; stride >>= 1) {
		__syncthreads();
		unsigned pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        if (compare<dir>(sInput[pos], sInput[pos + stride])) {
            swap(sInput[pos], sInput[pos + stride]);
        }
	}

    __syncthreads();
    input[2 * blockDim.x * blockIdx.x + threadIdx.x] = sInput[threadIdx.x];
    input[2 * blockDim.x * blockIdx.x + threadIdx.x + THREADS] = sInput[threadIdx.x + THREADS];
}


template <class T1, class T2>
void basic::merge::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    // dim3 gridDim(divideUp(SIZE, THREADS));
    // dim3 blockDim(THREADS);

    // mergeSortStep1<<<gridDim, blockDim>>>(inputs[DEVICE_INPUT]);
    // mergeSortStep2<<<gridDim, blockDim>>>(inputs[DEVICE_INPUT], outputs[DEVICE_OUTPUT], THREADS);

    dim3 gridDim(divideUp(SIZE, 2 * THREADS));
    dim3 gridDim2(divideUp(SIZE, 2 * THREADS) - 1);
    dim3 blockDim(THREADS);

    for (unsigned int idx = 0; idx < gridDim.x; ++idx) {
        mergeSortWithBinarySearch<DIRECTION><<<gridDim, blockDim>>>(inputs[DEVICE_INPUT]);
        mergeSortWithBinarySearch<DIRECTION><<<gridDim2, blockDim>>>(inputs[DEVICE_INPUT] + THREADS);
    }
    cudaDeviceSynchronize();

    checkCudaError(cudaGetLastError(), "Merge Sort launch failed - ");
    
    checkCudaError(cudaMemcpy(inputs[HOST_INPUT], outputs[DEVICE_OUTPUT], SIZE * sizeof(T2), cudaMemcpyDeviceToHost), "cudaMemcpy failed! (Device to Host) - ");
}

template void basic::merge::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);