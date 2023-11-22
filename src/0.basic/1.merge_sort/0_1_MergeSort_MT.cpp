#include "0_1_MergeSort_MT.h"
constexpr size_t THREAD_COUNT = 8;

void checkResult(TARGET_INPUT_TYPE* input, size_t start, size_t end) {
    for (size_t idx = start; idx < end; ++idx) {
        if (idx + 1 >= SIZE)
            return;

        if (input[idx] > input[idx + 1]) {
            throw std::runtime_error("Merge sort not complete");
        }
    }
}

int compLess( const void* lhs, const void* rhs ) {
	uint32_t lval = *(static_cast<const uint32_t*>(lhs));
	uint32_t rval = *(static_cast<const uint32_t*>(rhs));
	return (lval - rval);
}

template <class T1, class T2>
void basic::merge::run_comparison_target(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    std::vector<std::future<void>> futures;
    
    ThreadPool tp(THREAD_COUNT); 

    for (size_t t = 0; t < THREAD_COUNT; ++t) {
        size_t start = (SIZE / THREAD_COUNT) * t;
        size_t end = (t < THREAD_COUNT - 1) ? (SIZE / THREAD_COUNT) * (t + 1) : SIZE;

        futures.emplace_back(tp.addJob(checkResult, inputs[HOST_INPUT], start, end));
    }

    for (auto& f : futures) {
        f.get(); // for exception
    }
}

template void basic::merge::run_comparison_target(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);