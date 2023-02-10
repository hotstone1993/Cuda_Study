#include "0.basic/00_MatMul_Support.h"

#define RUN(NAMESPACE, INPUTS, OUTPUTS) { \
    initCUDA(); \
    EventTimer timer; \ 
    NAMESPACE::setup(INPUTS, OUTPUTS); \
    \
    timer.startTimer(); \
    NAMESPACE::run(INPUTS, OUTPUTS); \
    timer.stopTimer(); \
    timer.printElapsedTime(); \
    \
    timer.startTimer(); \
    NAMESPACE##_mt::run(INPUTS, OUTPUTS); \
    timer.stopTimer(); \
    timer.printElapsedTime(); \
    \
    NAMESPACE::destroy(INPUTS, OUTPUTS); \
}

#define DESTROY(NAMESPACE, INPUTS, OUTPUTS) { \
    NAMESPACE::destroy(INPUTS, OUTPUTS); \
    std::cerr << message.what() << std::endl; \
}