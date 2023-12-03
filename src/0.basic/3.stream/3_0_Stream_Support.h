
#ifndef STREAM_SETUP
#define STREAM_SETUP

#include "3_0_Null_Stream.cuh"
#include "3_0_Stream.cuh"

namespace basic::stream {
    void printInfo() {
        std::cerr << "[Stream VS Null-Stream]" << std::endl;
    }
    
    template <class T1>
    void initRandom(std::vector<T1*>& inputs) {
        srand(static_cast<unsigned int>(time(nullptr)));

        for (size_t idx = 0; idx < SIZE; ++idx) {
            T1 randomValue = rand() % (1 << 15);
            inputs[HOST_INPUT1][idx] = randomValue;
            inputs[HOST_INPUT2][idx] = randomValue;
        }
    }

    template <class T1, class T2>
    void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        inputs.resize(INPUT_COUNT);
        outputs.resize(OUTPUT_COUNT);

        inputs[HOST_INPUT1] = new T1[SIZE];
	    checkCudaError(cudaMallocHost(&inputs[HOST_INPUT2], sizeof(T1) * SIZE), "cudaMallocHost - ");
        CUDA_MALLOC(inputs[DEVICE_INPUT1], SIZE, T1)
        CUDA_MALLOC(inputs[DEVICE_INPUT2], SIZE, T1)

        outputs[HOST_OUTPUT1] = new T2[SIZE];
	    checkCudaError(cudaMallocHost(&outputs[HOST_OUTPUT2], sizeof(T2) * SIZE), "cudaMallocHost - ");
        CUDA_MALLOC(outputs[DEVICE_OUTPUT1], SIZE, T2)
        CUDA_MALLOC(outputs[DEVICE_OUTPUT2], SIZE, T2)

        initRandom(inputs);
    }
        
    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        delete[] inputs[HOST_INPUT1];
        cudaFreeHost(inputs[HOST_INPUT2]);
        cudaFree(inputs[DEVICE_INPUT1]);
        cudaFree(inputs[DEVICE_INPUT2]);

        delete[] outputs[HOST_OUTPUT1];
        cudaFreeHost(inputs[HOST_OUTPUT2]);
        cudaFree(outputs[DEVICE_OUTPUT1]);
        cudaFree(outputs[DEVICE_OUTPUT2]);
    }
}

#endif // STREAM_SETUP