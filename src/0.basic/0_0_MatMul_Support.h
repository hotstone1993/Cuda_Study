
#ifndef MUL_MATRIX_SETUP
#define MUL_MATRIX_SETUP

#include "0_0_MatMul_MT.h"
#include "0_0_MatMul.cuh"

namespace basic::matmul {
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
            throw std::runtime_error("cudaMemcpy failed! (Host to Device)");
        }
        cudaStatus = cudaMemcpy(inputs[DEVICE_INPUT2], inputs[HOST_INPUT2], SIZE * SIZE * sizeof(T1), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed! (Host to Device)");
        }
    }

    template <class T1, class T2>
    requires (std::is_same_v<T1, T2> && std::is_same_v<T2, T1>)
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
}

#endif // MUL_MATRIX_SETUP