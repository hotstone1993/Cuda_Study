#include "3_0_Stream.cuh"

template <class T1, class T2>
void basic::stream::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
}

template void basic::stream::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);