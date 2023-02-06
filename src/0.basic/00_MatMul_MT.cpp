#include "00_MatMul_MT.h"
#include "00_MatMul_Const.h"
#include "ThreadPool.h"
#include <iostream>

void func(const std::vector<TARGET_TYPE*>& inputs, std::vector<TARGET_TYPE*>& outputs, size_t startX, size_t startY, size_t endX, size_t endY) {
    
    for (size_t y = startY; y < endY; ++y) {
        for (size_t x = startX; x < endX; ++x) {
            int result = 0;
            size_t index = SIZE * y + x;

            for (size_t k = 0; k < SIZE; ++k) {
                result += (inputs[HOST_INPUT1][SIZE * y + k] * inputs[HOST_INPUT2][SIZE * k + x]);
            }

            if (outputs[HOST_OUTPUT1][index] != result) {
                std::cerr << "CUDA Result: " <<  outputs[HOST_OUTPUT1][index] << ", Result: " << result << std::endl;
                return;
            }
        }
    }
}

constexpr size_t THREAD_COUNT = 8;

template <class T1, class T2>
void basic_mt::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    ThreadPool tp(THREAD_COUNT); 
    
    for (size_t idx = 0; idx < THREAD_COUNT; ++idx) {
        size_t start = (SIZE / THREAD_COUNT) * idx;
        size_t end = (idx < THREAD_COUNT - 1) ? (SIZE / THREAD_COUNT) * (idx + 1) : SIZE;
        tp.addJob(func, inputs, outputs, start, start, end, end);
    }
}

template <class T1, class T2>
void basic_mt::destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
}

template void basic_mt::run(std::vector<TARGET_TYPE*>& inputs, std::vector<TARGET_TYPE*>& outputs);
template void basic_mt::destroy(std::vector<TARGET_TYPE*>& inputs, std::vector<TARGET_TYPE*>& outputs);