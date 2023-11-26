#include "3_0_Stream.cuh"

template <class T1, class T2>
void basic::stream::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    cudaStream_t stream[STREAMS];

    for (unsigned int i = 0; i < STREAMS; ++i) {
        cudaStreamCreate(&stream[i]);
    }
    
    unsigned int batchSize = (SIZE / STREAMS);

    for (unsigned int i = 0; i < STREAMS; ++i) {
        
    }
	cudaDeviceSynchronize();

    for (unsigned int i = 0; i < STREAMS; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

template void basic::stream::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);