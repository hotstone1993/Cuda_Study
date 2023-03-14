#include "0_2_LinearSearch_MT.h"

constexpr size_t THREAD_COUNT = 8;

void checkSearchResult(TARGET_INPUT_TYPE* input, size_t start, size_t end) {
    
}

template <class T1, class T2>
void basic::linear_search::run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    std::vector<std::future<void>> futures;
    
    ThreadPool tp(THREAD_COUNT); 

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        size_t start = (SIZE / THREAD_COUNT) * t;
        size_t end = (t < THREAD_COUNT - 1) ? (SIZE / THREAD_COUNT) * (t + 1) : SIZE;

        futures.emplace_back(tp.addJob(checkSearchResult, inputs[HOST_INPUT], start, end));
    }

    for (auto& f : futures) {
        f.get(); // for exception
    }
}

template void basic::linear_search::run_mt(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);