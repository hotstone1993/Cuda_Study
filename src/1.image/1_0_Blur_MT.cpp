#include "1_0_Blur_MT.h"

constexpr size_t THREAD_COUNT = 8;

template <class T1, class T2>
void image::blur::run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    ThreadPool tp(THREAD_COUNT); 
}

template void image::blur::run_mt(std::vector<TARGET_TYPE*>& inputs, std::vector<TARGET_TYPE*>& outputs);