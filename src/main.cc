#include <iostream>
#include "0.basic/00_MatMul_Support.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    std::vector<TARGET_TYPE*> inputs;
    std::vector<TARGET_TYPE*> outputs;

    try {
        initCUDA();
        RUN(basic, inputs, outputs)
    } catch(std::runtime_error& message) {
        basic::destroy(inputs, outputs);
        std::cerr << message.what() << std::endl;        
    }

    return 0;
}