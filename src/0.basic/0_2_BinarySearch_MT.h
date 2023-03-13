#ifndef BINARY_SEARCH_MT
#define BINARY_SEARCH_MT

#include "0_2_BinarySearch_Const.h"

namespace basic::binary_search {
    template <class T1, class T2>
    void run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // BINARY_SEARCH_MT