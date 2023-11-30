#include "3_0_Stream.cuh"

// Same as the algorithm of 3_0_Null_Stream.cu
__global__ void kernal2(TARGET_INPUT_TYPE* input, TARGET_OUTPUT_TYPE* output) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < gridDim.x * blockDim.x - 1) {
        output[idx] = (input[idx] + input[idx + 1]) / 2;
    } else {
        output[idx] = input[idx];
    }
}

template <class T1, class T2>
void basic::stream::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    cudaStream_t streams[STREAMS];

    for (unsigned int i = 0; i < STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    unsigned int batchSize = (SIZE / STREAMS);
    unsigned int remainingSize = SIZE + STREAMS - 1;
    unsigned int segment = divideUp(remainingSize, static_cast<unsigned int>(THREADS));

    for (unsigned int i = 0; i < STREAMS; ++i) {
        remainingSize -= segment;
        unsigned int realSize = segment;
        if (i == STREAMS - 1) {
            realSize = remainingSize;
        }
        dim3 gridDim(divideUp(realSize, static_cast<unsigned int>(THREADS)));
        dim3 blockDim(THREADS);
        T1* hostInput = inputs[HOST_INPUT2] + i * (segment - 1);
        T1* deviceInput = inputs[DEVICE_INPUT2] + i * (segment - 1);
        T2* hostOutput = outputs[HOST_OUTPUT2] + i * (segment - 1);
        T2* deviceOutput = outputs[DEVICE_OUTPUT2] + i * (segment - 1);

        cudaMemcpyAsync(deviceInput, hostInput, sizeof(T1) * segment, cudaMemcpyHostToDevice, streams[i]);

        kernal2<<<gridDim, blockDim, 0, streams[i]>>>(deviceInput, deviceOutput);

        cudaMemcpyAsync(hostOutput, deviceOutput, sizeof(T2) * segment, cudaMemcpyDeviceToHost, streams[i]);
    }
    checkCudaError(cudaGetLastError(), "Non-Null Stream launch failed - ");
	cudaDeviceSynchronize();

    for (unsigned int i = 0; i < STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

template void basic::stream::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);