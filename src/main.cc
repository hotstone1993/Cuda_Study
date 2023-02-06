#include <iostream>
#include "cuda_support.h"
#include "EventTimer.h"
#include "0.basic/00_MatMul.cuh"
#include "ThreadPool.h"

int main(int argc, char *argv[]) {
    std::vector<int*> inputs;
    std::vector<int*> outputs;

    try {
        EventTimer timer;
        initCUDA();
        timer.startTimer();

        basic::run(inputs, outputs);

        timer.stopTimer();
        timer.printElapsedTime();
    } catch(std::runtime_error& message) {
        basic::destroy(inputs, outputs);
        std::cerr << message.what() << std::endl;        
    }

    return 0;
}