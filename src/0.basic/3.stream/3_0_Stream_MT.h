#ifndef STREAM_MT
#define STREAM_MT

#include "3_0_Stream_Const.h"
#include "concept_utils.h"

namespace basic::stream {
    inline const char* getComparisonTarget() {
        return "Multi Thread";
    }
    template <class T1, class T2>
    void run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // STREAM_MT