#include "macro.h"

int main(int argc, char *argv[]) {
    std::vector<TARGET_TYPE*> inputs;
    std::vector<TARGET_TYPE*> outputs;

    try {
        RUN(inputs, outputs)
    } catch(std::runtime_error& message) {
        DESTROY(inputs, outputs)     
    }

    return 0;
}