#include "00_Mul_Matrix.cuh"

#define THREADS 32

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


void basic::run(float* a, float* b, float* c) {
}