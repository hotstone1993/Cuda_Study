#include <cmath>
#include <limits>
#include <iostream>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <stdexcept>


// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp) {
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
        // unless the result is subnormal
        || std::fabs(x-y) < std::numeric_limits<T>::min();
}


typedef int TARGET_TYPE;

#define RUN(NAMESPACE, INPUTS, OUTPUTS) { \
    EventTimer timer; \ 
    NAMESPACE::setup(INPUTS, OUTPUTS); \
    \
    timer.startTimer(); \
    timer.startTimer(); \
    NAMESPACE::run(INPUTS, OUTPUTS); \
    timer.stopTimer(); \
    timer.printElapsedTime(); \
    \
    timer.startTimer(); \
    timer.startTimer(); \
    NAMESPACE##_mt::run(INPUTS, OUTPUTS); \
    timer.stopTimer(); \
    timer.printElapsedTime(); \
    \
    NAMESPACE::destroy(INPUTS, OUTPUTS); \
}