#include "3_0_Stream.cuh"

// Same as the algorithm of 3_0_Null_Stream.cu
__global__ void kernal2(TARGET_INPUT_TYPE* input, TARGET_OUTPUT_TYPE* output, unsigned int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size - 1) {
        output[idx] = (input[idx] + input[idx + 1]) / 2;
    } else if (idx < size) {
        output[idx] = input[idx];    
    }
}

template <class T1, class T2>
void basic::stream::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    cudaStream_t streams[STREAMS];

    for (unsigned int i = 0; i < STREAMS; ++i) {
        checkCudaError(cudaStreamCreate(&streams[i]), "cudaStreamCreate - ");
    }
    
    unsigned int remainingSize = SIZE + STREAMS - 1;
    unsigned int segment = remainingSize / STREAMS;

    for (unsigned int i = 0; i < STREAMS; ++i) {
        unsigned int realSize = segment;
        if (i == STREAMS - 1) {
            realSize = remainingSize;
        }
        dim3 gridDim(divideUp(realSize, static_cast<unsigned int>(THREADS)));
        dim3 blockDim(THREADS);
        unsigned int offset = i * (segment - 1);

        T1* hostInput = inputs[HOST_INPUT2] + offset;
        T1* deviceInput = inputs[DEVICE_INPUT2] + offset;
        T2* hostOutput = outputs[HOST_OUTPUT2] + offset;
        T2* deviceOutput = outputs[DEVICE_OUTPUT2] + offset;

        checkCudaError(cudaMemcpyAsync(deviceInput, hostInput, sizeof(T1) * realSize, cudaMemcpyHostToDevice, streams[i]), "cudaMemcpyAsync(HostToDevice) - ");

        kernal2<<<gridDim, blockDim, 0, streams[i]>>>(deviceInput, deviceOutput, realSize);

        checkCudaError(cudaMemcpyAsync(hostOutput, deviceOutput, sizeof(T2) * realSize, cudaMemcpyDeviceToHost, streams[i]), "cudaMemcpyAsync(DeviceToHost) - ");
        
        remainingSize -= segment;
    }
	cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Non-Null Stream launch failed - ");

    for (unsigned int i = 0; i < STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

template void basic::stream::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);