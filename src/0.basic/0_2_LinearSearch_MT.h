#ifndef LINEAR_SEARCH_MT
#define LINEAR_SEARCH_MT

#include "0_2_LinearSearch_Const.h"

namespace basic::linear_search {
    template <class T1, class T2>
    void run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // LINEAR_SEARCH_MT