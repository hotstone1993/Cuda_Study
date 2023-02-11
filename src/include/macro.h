#include "0.basic/00_MatMul_Support.h"

#define TARGET_NAMESPACE basic

#define RUN(INPUTS, OUTPUTS) { \
    initCUDA(); \
    EventTimer timer; \ 
    TARGET_NAMESPACE::setup(INPUTS, OUTPUTS); \
    \
    timer.startTimer(); \
    TARGET_NAMESPACE::run(INPUTS, OUTPUTS); \
    timer.stopTimer(); \
    timer.printElapsedTime(); \
    \
    timer.startTimer(); \
    TARGET_NAMESPACE::run_mt(INPUTS, OUTPUTS); \
    timer.stopTimer(); \
    timer.printElapsedTime(); \
    \
    TARGET_NAMESPACE::destroy(INPUTS, OUTPUTS); \
}

#define DESTROY(INPUTS, OUTPUTS) { \
    TARGET_NAMESPACE::destroy(INPUTS, OUTPUTS); \
    std::cerr << message.what() << std::endl; \
}