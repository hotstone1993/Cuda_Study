#include <iostream>

#include "cuda_support.h"

int main(int argc, char *argv[]) {
    try {
        initCUDA();

        
    } catch(const char* message) {
        std::cout << message << std::endl;        
    }

    return 0;
}