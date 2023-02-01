#include <iostream>
#include "cuda_support.h"
#include "EventTimer.h"

int main(int argc, char *argv[]) {
    try {
        EventTimer timer;
        initCUDA();
        timer.startTimer();

        
        timer.stopTimer();
        timer.printElapsedTime();
    } catch(const char* message) {
        std::cerr << message << std::endl;        
    }

    return 0;
}