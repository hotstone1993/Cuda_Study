#include "0_1_MergeSort_MT.h"

constexpr size_t THREAD_COUNT = 8;

template <class T1, class T2>
void basic::merge::run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    std::vector<std::future<void>> futures;
    
    ThreadPool tp(THREAD_COUNT); 

    for (auto& f : futures) {
        f.get(); // for exception
    }
}

template void basic::merge::run_mt(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);