#ifndef MERGE_SORT_MT
#define MERGE_SORT_MT

#include "0_1_MergeSort_Const.h"

namespace basic::merge {
    template <class T1, class T2>
    void run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // MERGE_SORT_MT