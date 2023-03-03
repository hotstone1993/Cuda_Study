
#ifndef MERGE_SORT_SETUP
#define MERGE_SORT_SETUP

#include "0_1_MergeSort_MT.h"
#include "0_1_MergeSort.cuh"

namespace basic::merge {
    template <class T1>
    void initRandom(std::vector<T1*>& inputs) {
        srand(static_cast<unsigned int>(time(nullptr)));

        for (size_t idx = 0; idx < SIZE; ++idx) {
            inputs[HOST_INPUT][idx] = rand();
        }
    }

    template <class T1>
    void copyInputs(std::vector<T1*>& inputs) {
        checkCudaError(cudaMemcpy(inputs[DEVICE_INPUT], inputs[HOST_INPUT], SIZE * sizeof(T1), cudaMemcpyHostToDevice), "cudaMemcpy failed! (Host to Device) - ");
    }

    template <class T1, class T2>
    requires (std::is_same_v<T1, T2> && std::is_same_v<T2, T1>)
    void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        inputs.resize(INPUT_COUNT);

        inputs[HOST_INPUT] = new T1[SIZE];
        CUDA_MALLOC(inputs[DEVICE_INPUT], SIZE, T1)

        initRandom(inputs);
        copyInputs(inputs);
    }
        
    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        delete[] inputs[HOST_INPUT];
        cudaFree(inputs[DEVICE_INPUT]);
    }
}

#endif // MERGE_SORT_SETUP