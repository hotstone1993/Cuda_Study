#ifndef MUL_MATRIX_MT
#define MUL_MATRIX_MT

#include "00_MatMul_Const.h"

namespace basic::matmul {
    template <class T1, class T2>
    void run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // MUL_MATRIX_MT