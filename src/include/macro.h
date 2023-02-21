#include "0.basic/0_0_MatMul_Support.h"
// #include "1.image/1_0_Blur_Support.h"

#define TARGET_NAMESPACE basic::matmul

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