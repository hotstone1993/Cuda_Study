#include "1_0_Blur.cuh"

template <class T1, class T2>
void image::blur::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {}

template void image::blur::run(std::vector<TARGET_TYPE*>& inputs, std::vector<TARGET_TYPE*>& outputs);