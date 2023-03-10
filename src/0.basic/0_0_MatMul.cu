#include "0_0_MatMul.cuh"

namespace cg = cooperative_groups;

__global__ void mulMatrix(TARGET_OUTPUT_TYPE* c, const TARGET_INPUT_TYPE* a, const TARGET_INPUT_TYPE* b, const unsigned int N)
{
    cg::thread_block block = cg::this_thread_block();
    
    int col = block.dim_threads().x * block.group_index().x + block.thread_index().x;
    int row = block.dim_threads().y * block.group_index().y + block.thread_index().y;
    
    if (row >= N || col >= N)
        return;
    
    TARGET_OUTPUT_TYPE sum = 0;

    for (unsigned int idx = 0; idx < N; ++idx) {
        sum += (a[N * row + idx] * b[idx * N + col]);
    }
    c[row * N + col] = sum;
}


__global__ void mulMatrixWithSharedMemory(TARGET_OUTPUT_TYPE* c, const TARGET_INPUT_TYPE* a, const TARGET_INPUT_TYPE* b, const unsigned int N)
{
    cg::thread_block block = cg::this_thread_block();

    __shared__ TARGET_INPUT_TYPE tempA[THREADS][THREADS];
    __shared__ TARGET_INPUT_TYPE tempB[THREADS][THREADS];

    int col = block.dim_threads().x * block.group_index().x + block.thread_index().x;
    int row = block.dim_threads().y * block.group_index().y + block.thread_index().y;

    int localCol = block.thread_index().x;
    int localRow = block.thread_index().y;

    TARGET_OUTPUT_TYPE sum = 0;

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

        block.sync();

        for (unsigned int idx = 0; idx < blockDim.x; ++idx) {
            sum += (tempA[localRow][idx] * tempB[idx][localCol]);
        }

        block.sync();
    }

    if (row >= N || col >= N)
        return;

    c[row * N + col] += sum;
}



template <class T1, class T2>
void basic::matmul::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    int face = (SIZE + THREADS - 1) / THREADS;
    dim3 gridDim(face, face);
    dim3 blockDim(THREADS, THREADS);

    mulMatrixWithSharedMemory<<<gridDim, blockDim>>>(outputs[DEVICE_OUTPUT1], inputs[DEVICE_INPUT1], inputs[DEVICE_INPUT2], SIZE);

    checkCudaError(cudaGetLastError(), "MatMul launch failed - ");
    
    checkCudaError(cudaMemcpy(outputs[HOST_OUTPUT1], outputs[DEVICE_OUTPUT1], SIZE * SIZE * sizeof(T2), cudaMemcpyDeviceToHost), "cudaMemcpy failed! (Device to Host) - ");
}

template void basic::matmul::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);