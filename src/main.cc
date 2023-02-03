#include <iostream>
#include "cuda_support.h"
#include "EventTimer.h"
#include "0.basic/00_MatMul.cuh"

int main(int argc, char *argv[]) {
    float* host_input, *device_input = nullptr;
    float* host_output, *device_output = nullptr;

    try {
        EventTimer timer;
        initCUDA();
        timer.startTimer();
        CUDA_ALLOC(1024, float, 1024, float)

        basic::run(device_input, device_output, device_output);

        CUDA_DEALLOC()
        timer.stopTimer();
        timer.printElapsedTime();
    } catch(const char* message) {
        std::cerr << message << std::endl;        
    }

    return 0;
}