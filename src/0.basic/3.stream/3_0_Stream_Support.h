
#ifndef STREAM_SETUP
#define STREAM_SETUP

#include "3_0_Null_Stream.cuh"
#include "3_0_Stream.cuh"

namespace basic::stream {
    void printInfo() {
        std::cerr << "Stream" << std::endl;
    }
    
    template <class T1>
    void initRandom(std::vector<T1*>& inputs) {
        srand(static_cast<unsigned int>(time(nullptr)));

        for (size_t idx = 0; idx < SIZE; ++idx) {
            inputs[HOST_INPUT1][idx] = rand() % (1 << 15);
        }
    }

    template <class T1>
    void copyInputs(std::vector<T1*>& inputs) {
        checkCudaError(cudaMemcpy(inputs[DEVICE_INPUT1], inputs[HOST_INPUT1], SIZE * sizeof(T1), cudaMemcpyHostToDevice), "cudaMemcpy failed! (Host to Device) - ");
    }

    template <class T1, class T2>
    void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        inputs.resize(INPUT_COUNT);
        outputs.resize(OUTPUT_COUNT);

        inputs[HOST_INPUT1] = new T1[SIZE];
        CUDA_MALLOC(inputs[DEVICE_INPUT1], SIZE, T1)

        outputs[HOST_OUTPUT1] = new T2[SIZE];
        CUDA_MALLOC(outputs[DEVICE_OUTPUT1], SIZE, T2)

        initRandom(inputs);
        copyInputs(inputs);
    }
        
    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        delete[] inputs[HOST_INPUT1];
        cudaFree(inputs[DEVICE_INPUT1]);

        delete[] outputs[HOST_OUTPUT1];
        cudaFree(outputs[DEVICE_OUTPUT1]);
    }
}

#endif // STREAM_SETUP