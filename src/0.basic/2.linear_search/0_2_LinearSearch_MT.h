#ifndef LINEAR_SEARCH_MT
#define LINEAR_SEARCH_MT

#include "0_2_LinearSearch_Const.h"
#include "concept_utils.h"

namespace basic::linear_search {
    inline const char* getComparisonTarget() {
        return "Just Check";
    }
    template <class T1, class T2>
    void run_comparison_target(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // LINEAR_SEARCH_MT