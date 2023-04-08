#include "0_1_MergeSort.cuh"

namespace cg = cooperative_groups;

// Nvidia Sample Code
#define W (sizeof(uint) * 8)
static inline __device__ uint nextPowerOfTwo(uint x) {
  /*
      --x;
      x |= x >> 1;
      x |= x >> 2;
      x |= x >> 4;
      x |= x >> 8;
      x |= x >> 16;
      return ++x;
  */
  return 1U << (W - __clz(x - 1));
}

__device__ void naiveMerge(TARGET_INPUT_TYPE* input, TARGET_INPUT_TYPE* temp, unsigned int start, unsigned int idx, unsigned int stride, unsigned int size) {
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
__global__ void naiveMergeSortStep1(TARGET_INPUT_TYPE* input)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ TARGET_INPUT_TYPE temp[THREADS];
    unsigned int stride = 2;

    while (stride < THREADS) {
        if (idx % stride == 0) {
            naiveMerge(input, temp, idx, threadIdx.x, stride, THREADS);
        }
        __syncthreads();
        if (idx < SIZE) {
            input[idx] = temp[threadIdx.x];
        }
        __syncthreads();
        stride <<= 1;
    }
}

__global__ void naiveMergeSortStep2(TARGET_INPUT_TYPE* input, TARGET_OUTPUT_TYPE* output, unsigned int stride)
{
    bool flag = true;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    TARGET_INPUT_TYPE* buffer1 = input;
    TARGET_INPUT_TYPE* buffer2 = output;

    while (stride / 2 < SIZE) {
        if (idx % stride == 0) {
            if (flag) {
                naiveMerge(buffer1, buffer2, idx, idx, stride, SIZE);
            } else {
                naiveMerge(buffer2, buffer1, idx, idx, stride, SIZE);
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
        unsigned int posA = getInclusivePositionByBinarySearch<dir>(base + stride, valueA, stride, stride) + basePos;
        unsigned int posB = getExclusivePositionByBinarySearch<dir>(base, valueB, stride, stride) + basePos;
        
        __syncthreads();
        base[posA] = valueA;
        base[posB] = valueB;
    }

    __syncthreads();
    input[idx] = temp[threadIdx.x];
    input[blockDim.x + idx] = temp[blockDim.x + threadIdx.x];
}

template <bool dir>
__global__ void getRanks(TARGET_INPUT_TYPE* input, TARGET_INPUT_TYPE* rankA, TARGET_INPUT_TYPE* rankB, unsigned int stride, unsigned int rankCount) {
    // Two rank per thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rankCount <= idx) {
        return;
    }
        
    const unsigned int baseIdx = idx & ((stride / SAMPLE_STRIDE) - 1);
    const unsigned int segmentBase = (idx - baseIdx) * (2 * SAMPLE_STRIDE);
    input += segmentBase;
    rankA += segmentBase / SAMPLE_STRIDE;
    rankB += segmentBase / SAMPLE_STRIDE;

    const unsigned int rankACount = stride % SAMPLE_STRIDE != 0 ? stride / SAMPLE_STRIDE + 1 : stride / SAMPLE_STRIDE;
    const unsigned int rankBElements = umin(stride, SIZE - segmentBase - stride);
    const unsigned int rankBCount = rankBElements % SAMPLE_STRIDE != 0 ? rankBElements / SAMPLE_STRIDE + 1 : rankBElements / SAMPLE_STRIDE;

    if (idx < rankACount) {
        rankA[baseIdx] = SAMPLE_STRIDE * baseIdx;
        unsigned int pos = getExclusivePositionByBinarySearch<dir>(input + stride, input[SAMPLE_STRIDE * baseIdx], rankBElements, nextPowerOfTwo(rankBElements));
        rankB[baseIdx] = pos;
    }

    if (idx < rankBCount) {
        rankB[baseIdx + (stride / SAMPLE_STRIDE)] = SAMPLE_STRIDE * baseIdx;
        unsigned int pos = getExclusivePositionByBinarySearch<dir>(input, input[stride + SAMPLE_STRIDE * baseIdx], SAMPLE_STRIDE, nextPowerOfTwo(SAMPLE_STRIDE));
        rankA[baseIdx + (stride / SAMPLE_STRIDE)] = pos;
    }
}

template <bool dir>
__global__ void getLimits(TARGET_INPUT_TYPE* rank, TARGET_INPUT_TYPE* limits, unsigned int stride, unsigned int rankCount) {
    // Two rank per thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rankCount <= idx) {
        return;
    }

    const unsigned int baseIdx = idx & ((stride / SAMPLE_STRIDE) - 1);
    const unsigned int segmentBase = (idx - baseIdx) * (2 * SAMPLE_STRIDE);
    rank += ((idx - baseIdx) * 2);
    limits += ((idx - baseIdx) * 2);

    const unsigned int rankACount = stride % SAMPLE_STRIDE != 0 ? stride / SAMPLE_STRIDE + 1 : stride / SAMPLE_STRIDE;
    const unsigned int rankBElements = umin(stride, SIZE - segmentBase - stride);
    const unsigned int rankBCount = rankBElements % SAMPLE_STRIDE != 0 ? rankBElements / SAMPLE_STRIDE + 1 : rankBElements / SAMPLE_STRIDE;

    if (idx < rankACount) {
        unsigned int pos = getExclusivePositionByBinarySearch<dir>(rank + rankACount, rank[baseIdx], rankBCount, nextPowerOfTwo(rankBCount)) + baseIdx;
        limits[pos] = rank[baseIdx];
    }

    if (idx < rankBCount) {
        unsigned int pos = getExclusivePositionByBinarySearch<dir>(rank, rank[rankACount + baseIdx], rankACount, nextPowerOfTwo(rankACount)) + baseIdx;
        limits[pos] = rank[rankACount + baseIdx];
    }
}


template <bool dir>
__device__ void merge(TARGET_INPUT_TYPE* output, TARGET_INPUT_TYPE* intputA, TARGET_INPUT_TYPE* intputB, uint lenA, uint nPowTwoLenA, uint lenB, uint nPowTwoLenB, cg::thread_block cta) {
    TARGET_INPUT_TYPE a, b, dstPosA, dstPosB;

    if (threadIdx.x < lenA) {
        a = intputA[threadIdx.x];
        dstPosA = getExclusivePositionByBinarySearch<dir>(intputB, a, lenB, nPowTwoLenB) + threadIdx.x;
    }

    if (threadIdx.x < lenB) {
        b = intputB[threadIdx.x];
        dstPosB = getInclusivePositionByBinarySearch<dir>(intputA, b, lenA, nPowTwoLenA) + threadIdx.x;
    }

    cg::sync(cta);

    if (threadIdx.x < lenA) {
        output[dstPosA] = a;
    }

    if (threadIdx.x < lenB) {
        output[dstPosB] = b;
    }
}

template <uint dir>
__global__ void mergeSort(TARGET_INPUT_TYPE* output, TARGET_INPUT_TYPE* input, TARGET_INPUT_TYPE* limitsA, TARGET_INPUT_TYPE* limitsB, uint stride) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ TARGET_INPUT_TYPE temp[2 * SAMPLE_STRIDE];

  const uint intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
  const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
  input += segmentBase;
  output += segmentBase;

  __shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

  if (threadIdx.x == 0) {
    uint segmentElementsA = stride;
    uint segmentElementsB = umin(stride, SIZE - segmentBase - stride);
    uint segmentSamplesA = segmentElementsA % SAMPLE_STRIDE != 0 ? segmentElementsA / SAMPLE_STRIDE + 1 : segmentElementsA / SAMPLE_STRIDE;
    uint segmentSamplesB = segmentElementsB % SAMPLE_STRIDE != 0 ? segmentElementsB / SAMPLE_STRIDE + 1 : segmentElementsB / SAMPLE_STRIDE;
    uint segmentSamples = segmentSamplesA + segmentSamplesB;

    startSrcA = limitsA[blockIdx.x];
    startSrcB = limitsB[blockIdx.x];
    TARGET_INPUT_TYPE endSrcA = (intervalI + 1 < segmentSamples) ? limitsA[blockIdx.x + 1] : segmentElementsA;
    TARGET_INPUT_TYPE endSrcB = (intervalI + 1 < segmentSamples) ? limitsB[blockIdx.x + 1] : segmentElementsB;
    lenSrcA = endSrcA - startSrcA;
    lenSrcB = endSrcB - startSrcB;
    startDstA = startSrcA + startSrcB;
    startDstB = startDstA + lenSrcA;
  }

  cg::sync(cta);
  if (threadIdx.x < lenSrcA) {
    temp[threadIdx.x + 0] = input[0 + startSrcA + threadIdx.x];
  }

  if (threadIdx.x < lenSrcB) {
    temp[threadIdx.x + SAMPLE_STRIDE] = input[stride + startSrcB + threadIdx.x];
  }

  cg::sync(cta);
  merge<dir>(temp, temp + 0, temp + SAMPLE_STRIDE, lenSrcA, SAMPLE_STRIDE, lenSrcB, SAMPLE_STRIDE, cta);
  cg::sync(cta);

  if (threadIdx.x < lenSrcA) {
    output[startDstA + threadIdx.x] = temp[threadIdx.x];
  }

  if (threadIdx.x < lenSrcB) {
    output[startDstB + threadIdx.x] = temp[lenSrcA + threadIdx.x];
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

    // buble sort + merge sort
    // dim3 gridDim(divideUp(SIZE, 2 * THREADS));
    // dim3 gridDim2(divideUp(SIZE, 2 * THREADS) - 1);
    // dim3 blockDim(THREADS);

    // for (unsigned int idx = 0; idx < gridDim.x; ++idx) {
    //     mergeSortWithBinarySearch<DIRECTION><<<gridDim, blockDim>>>(inputs[DEVICE_INPUT]);
    //     mergeSortWithBinarySearch<DIRECTION><<<gridDim2, blockDim>>>(inputs[DEVICE_INPUT] + THREADS);
    // }

    dim3 gridDim(divideUp(SIZE, 2 * THREADS));
    dim3 blockDim(THREADS);
    mergeSortWithBinarySearch<DIRECTION><<<gridDim, blockDim>>>(inputs[DEVICE_INPUT]);

    TARGET_INPUT_TYPE* input = inputs[DEVICE_INPUT];
    TARGET_INPUT_TYPE* output = outputs[DEVICE_OUTPUT];

    for (unsigned int stride = 2 * THREADS; stride < SIZE; stride <<= 1) {
        // step 1
        unsigned int lastSegmentSize = SIZE % (2 * stride);
        unsigned int rankCount = lastSegmentSize > stride ? (SIZE + 2 * stride - lastSegmentSize) / (2 * SAMPLE_STRIDE)
        : (SIZE - lastSegmentSize) / (2 * SAMPLE_STRIDE);
        blockDim = make_uint3(256, 1, 1);
        gridDim = make_uint3(divideUp(rankCount, 256U), 1, 1);

        getRanks<DIRECTION><<<gridDim, blockDim>>>(input, inputs[DEVICE_RANK_A], inputs[DEVICE_RANK_B], stride, rankCount);
        checkCudaError(cudaGetLastError(), "getRanks - ");

        // step 2
        blockDim = make_uint3(256, 1, 1);
        gridDim = make_uint3(divideUp(rankCount, 256U), 1, 1);

        getLimits<DIRECTION><<<gridDim, blockDim>>>(inputs[DEVICE_RANK_A], inputs[DEVICE_LIMITS_A], stride, rankCount);
        checkCudaError(cudaGetLastError(), "getLimits A failed - ");
        getLimits<DIRECTION><<<gridDim, blockDim>>>(inputs[DEVICE_RANK_B], inputs[DEVICE_LIMITS_B], stride, rankCount);
        checkCudaError(cudaGetLastError(), "getLimits B failed - ");

        mergeSort<DIRECTION><<<gridDim, blockDim>>>(output, input, inputs[DEVICE_LIMITS_A], inputs[DEVICE_LIMITS_B], stride);

        TARGET_INPUT_TYPE* temp = input;
        input = output;
        output = temp;
    }

    cudaDeviceSynchronize();

    checkCudaError(cudaGetLastError(), "Merge Sort launch failed - ");
    
    checkCudaError(cudaMemcpy(inputs[HOST_INPUT], output, SIZE * sizeof(T2), cudaMemcpyDeviceToHost), "cudaMemcpy failed! (Device to Host) - ");
}

template void basic::merge::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);