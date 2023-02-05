#include "00_MatMul.cuh"
#include "cuda_support.h"

__global__ void mulMatrix(int* c, const int* a, const int* b, const unsigned int N)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int sum = 0;
    if (row >= N && col >= N)
        return;

    for (unsigned int idx = 0; idx < N; ++idx) {
        sum += (a[row * N + idx] * b[N * idx + col]);
    }
    c[row * N + col] = sum;
}


__global__ void mulMatrixWithSharedMemory(int* c, const int* a, const int* b, const unsigned int N)
{
    __shared__ int  tempA[THREADS][THREADS];
    __shared__ int  tempB[THREADS][THREADS];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    int localCol = threadIdx.x;
    int localRow = threadIdx.y;

    int sum = 0;

    for (unsigned int bid = 0; bid < ceil((float)N / blockDim.x); ++bid) {
        if (row < N && bid * blockDim.x + localCol < N) {
            tempA[localRow][localCol] = a[N * row + (bid * blockDim.x + localCol)];
        }
        else {
            tempA[localRow][localCol] = 0;
        }

        if (col < N && bid * blockDim.y + localRow < N) {
            tempB[localRow][localCol] = b[N * (bid * blockDim.y + localRow) + col];
        }
        else {
            tempB[localRow][localCol] = 0;
        }

        __syncthreads();

        for (unsigned int idx = 0; idx < blockDim.x; ++idx) {
            sum += (tempA[localRow][idx] * tempB[idx][localCol]);
        }

        __syncthreads();
    }

    if (row >= N && col >= N)
        return;

    c[row * N + col] = sum;
}


#define HOST_INPUT1 0
#define HOST_INPUT2 1
#define DEVICE_INPUT1 2
#define DEVICE_INPUT2 3
#define INPUT_COUNT 4

#define HOST_OUTPUT1 0
#define DEVICE_OUTPUT1 1
#define OUTPUT_COUNT 2

template <class T1>
void initRandom(std::vector<T1*>& inputs) {
    srand(static_cast<unsigned int>(time(nullptr)));

    for (size_t idx = 0; idx < SIZE * SIZE; ++idx) {
        inputs[HOST_INPUT1][idx] = rand() % 21 - 10;
        inputs[HOST_INPUT2][idx] = rand() % 21 - 10;
    }
}

template <class T1>
void copyInputs(std::vector<T1*>& inputs) {
    cudaError_t cudaStatus = cudaMemcpy(inputs[DEVICE_INPUT1], inputs[HOST_INPUT1], SIZE * SIZE * sizeof(T1), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw "cudaMemcpy failed! (Host to Device)";
    }
    cudaStatus = cudaMemcpy(inputs[DEVICE_INPUT2], inputs[HOST_INPUT2], SIZE * SIZE * sizeof(T1), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        throw "cudaMemcpy failed! (Host to Device)";
    }
}

template <class T1, class T2>
void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    inputs.resize(INPUT_COUNT);
    outputs.resize(OUTPUT_COUNT);

    inputs[HOST_INPUT1] = new T1[SIZE * SIZE];
    inputs[HOST_INPUT2] = new T1[SIZE * SIZE];
    CUDA_MALLOC(inputs[DEVICE_INPUT1], SIZE * SIZE, T1)
    CUDA_MALLOC(inputs[DEVICE_INPUT2], SIZE * SIZE, T1)

    outputs[HOST_OUTPUT1] = new T2[SIZE * SIZE];
    CUDA_MALLOC(outputs[DEVICE_OUTPUT1], SIZE * SIZE, T2)

    initRandom(inputs);
    copyInputs(inputs);
}

template <class T1, class T2>
void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    delete[] inputs[HOST_INPUT1];
    delete[] inputs[HOST_INPUT2];
    cudaFree(inputs[DEVICE_INPUT1]);
    cudaFree(inputs[DEVICE_INPUT2]);

    delete[] outputs[HOST_OUTPUT1];
    cudaFree(outputs[DEVICE_OUTPUT1]);
}

template <class T1, class T2>
void basic::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    setup(inputs, outputs);

    int face = (SIZE + THREADS - 1) / THREADS;
    dim3 gridDim(face, face);
    dim3 blockDim(THREADS, THREADS);

    mulMatrixWithSharedMemory<<<gridDim, blockDim>>>(inputs[DEVICE_INPUT1], inputs[DEVICE_INPUT2], outputs[DEVICE_OUTPUT1], SIZE);
    cudaDeviceSynchronize();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw "MatMul launch failed: %s\n";
    }
    
    cudaStatus = cudaMemcpy(outputs[HOST_OUTPUT1], outputs[DEVICE_OUTPUT1], SIZE * SIZE * sizeof(T2), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw "cudaMemcpy failed! (Device to Host)";
    }

    destroy(inputs, outputs);
}

template void basic::run(std::vector<int*>& inputs, std::vector<int*>& outputs);