#include "target.h"

#define RUN(INPUTS, OUTPUTS) { \
    initCUDA(); \
    EventTimer timer; \
    TARGET_NAMESPACE::printInfo(); \
    \
    timer.startTimer(); \
    TARGET_NAMESPACE::setup(INPUTS, OUTPUTS); \
    timer.stopTimer(); \
    timer.printElapsedTime("Setup"); \
    \
    timer.startTimer(); \
    TARGET_NAMESPACE::run(INPUTS, OUTPUTS); \
    timer.stopTimer(); \
    timer.printElapsedTime("CUDA"); \
    \
    timer.startTimer(); \
    TARGET_NAMESPACE::run_comparison_target(INPUTS, OUTPUTS); \
    timer.stopTimer(); \
    timer.printElapsedTime(TARGET_NAMESPACE::getComparisonTarget()); \
    \
    TARGET_NAMESPACE::destroy(INPUTS, OUTPUTS); \
}

#define DESTROY(INPUTS, OUTPUTS) { \
    TARGET_NAMESPACE::destroy(INPUTS, OUTPUTS); \
    std::cerr << message.what() << std::endl; \
}