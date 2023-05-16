
#ifndef LINEAR_SEARCH_SETUP
#define LINEAR_SEARCH_SETUP

#include "0_2_LinearSearch_MT.h"
#include "0_2_LinearSearch.cuh"

namespace basic::linear_search {
    void printInfo() {
        std::cerr << "Linear Search" << std::endl;
    }

    template <class T1, class T2>
    void copyInputs(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        checkCudaError(cudaMemcpy(inputs[DEVICE_INPUT], inputs[HOST_INPUT], SIZE * sizeof(T1), cudaMemcpyHostToDevice), "cudaMemcpy failed! (Host to Device) - ");
        checkCudaError(cudaMemset(outputs[DEVICE_OUTPUT], SIZE, sizeof(T2)), "cudaMemset failed! - ");
    }

    template <class T1, class T2>
    void initRandom(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        srand(static_cast<unsigned int>(time(nullptr)));

        for (size_t idx = 0; idx < SIZE; ++idx) {
            inputs[HOST_INPUT][idx] = std::abs(rand()) % SIZE;
        }

        while(true) {
            int target = std::abs(rand()) % SIZE;

            for (size_t idx = 0; idx < SIZE; ++idx) {
                if (inputs[HOST_INPUT][idx] == target) {
                    inputs[TARGET_NUMBER][0] = target;
                    inputs[TARGET_NUMBER][1] = idx;
                    goto afterLoop;
                }
            }
        }
    afterLoop:
        copyInputs(inputs, outputs);
    }

    template <class T1, class T2>
    requires (std::is_same_v<T1, T2> && std::is_same_v<T2, T1>)
    void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        inputs.resize(INPUT_COUNT);
        outputs.resize(OUTPUT_COUNT);

        inputs[HOST_INPUT] = new T1[SIZE];
        inputs[TARGET_NUMBER] = new T1[2];
        outputs[HOST_OUTPUT] = new T2();
        CUDA_MALLOC(inputs[DEVICE_INPUT], SIZE, T1)
        CUDA_MALLOC(outputs[DEVICE_OUTPUT], 1, T2)

        initRandom(inputs, outputs);
    }
        
    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        delete[] inputs[HOST_INPUT];
        delete[] inputs[TARGET_NUMBER];
        delete outputs[HOST_OUTPUT];
        cudaFree(inputs[DEVICE_INPUT]);
        cudaFree(outputs[DEVICE_OUTPUT]);
    }
}

#endif // LINEAR_SEARCH_SETUP