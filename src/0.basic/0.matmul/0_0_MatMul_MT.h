#ifndef MUL_MATRIX_MT
#define MUL_MATRIX_MT

#include "0_0_MatMul_Const.h"

namespace basic::matmul {
    inline const char* getComparisonTarget() {
        return "Muti Thread";
    }
    template <class T1, class T2>
    void run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // MUL_MATRIX_MT