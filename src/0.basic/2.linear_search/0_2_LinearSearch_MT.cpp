#include "0_2_LinearSearch_MT.h"

constexpr size_t THREAD_COUNT = 8;

template <class T1, class T2>
void basic::linear_search::run_comparison_target(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    // just check result
    if (inputs[TARGET_NUMBER][1] != *outputs[HOST_OUTPUT]) {
        std::cout << "answer(" << inputs[TARGET_NUMBER][0] << "): " << inputs[TARGET_NUMBER][1] << ",  output: " << *outputs[HOST_OUTPUT] << std::endl;
    }
}

template void basic::linear_search::run_comparison_target(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);