#ifndef MERGE_SORT_MT
#define MERGE_SORT_MT

#include "0_1_MergeSort_Const.h"
#include "concept_utils.h"

namespace basic::merge {
    inline const char* getComparisonTarget() {
        return "just check";
    }
    template <class T1, class T2>
    void run_comparison_target(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // MERGE_SORT_MT