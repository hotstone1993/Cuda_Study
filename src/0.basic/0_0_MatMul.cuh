#ifndef MUL_MATRIX
#define MUL_MATRIX

#include "0_0_MatMul_Const.h"

namespace basic::matmul {
    template <class T1, class T2>
    void run(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif
