#include "0_0_MatMul_MT.h"

void func(const std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs, size_t startX, size_t startY, size_t endX, size_t endY) {
    for (size_t y = startY; y < endY; ++y) {
        for (size_t x = startX; x < endX; ++x) {
            TARGET_OUTPUT_TYPE result = 0;
            size_t index = SIZE * y + x;

            for (size_t k = 0; k < SIZE; ++k) {
                result += (inputs[HOST_INPUT1][SIZE * y + k] * inputs[HOST_INPUT2][SIZE * k + x]);
            }

            if (outputs[HOST_OUTPUT1][index] != result) {
                std::string errorString = "CUDA Result: ";
                errorString += std::to_string(outputs[HOST_OUTPUT1][index]);
                errorString += ", result: ";
                errorString += std::to_string(result);

                throw std::runtime_error(errorString);
            }
        }
    }
}

constexpr size_t THREAD_COUNT = 8;

template <class T1, class T2>
void basic::matmul::run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    std::vector<std::future<void>> futures;
    
    ThreadPool tp(THREAD_COUNT); 
    
    for (size_t y = 0; y < THREAD_COUNT; ++y) {
        size_t yStart = (SIZE / THREAD_COUNT) * y;
        size_t yEnd = (y < THREAD_COUNT - 1) ? (SIZE / THREAD_COUNT) * (y + 1) : SIZE;
        for (size_t x = 0; x < THREAD_COUNT; ++x) { 
            size_t xStart = (SIZE / THREAD_COUNT) * x;
            size_t xEnd = (x < THREAD_COUNT - 1) ? (SIZE / THREAD_COUNT) * (x + 1) : SIZE;
            
            futures.emplace_back(tp.addJob(func, inputs, outputs, xStart, yStart, xEnd, yEnd));
        }
    }

    for (auto& f : futures) {
        f.get(); // for exception
    }
}

template void basic::matmul::run_mt(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);