#include "3_0_Stream_MT.h"

void func(const std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs, size_t startX, size_t startY, size_t endX, size_t endY) {
}

constexpr size_t THREAD_COUNT = 8;

template <class T1, class T2>
void basic::stream::run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    std::vector<std::future<void>> futures;
    
    ThreadPool tp(THREAD_COUNT); 

    for (auto& f : futures) {
        f.get(); // for exception
    }
}

template void basic::stream::run_mt(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);