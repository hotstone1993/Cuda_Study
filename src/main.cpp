#include "macro.h"

int main(int argc, char *argv[]) {
    std::vector<TARGET_INPUT_TYPE*> inputs;
    std::vector<TARGET_OUTPUT_TYPE*> outputs;

    try {
        RUN(inputs, outputs)
    } catch(std::runtime_error& message) {
        DESTROY(inputs, outputs)     
    } catch(std::string& message) {
        std::cerr << message << std::endl;
    }

    return 0;
}