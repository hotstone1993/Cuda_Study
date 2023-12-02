#ifndef NOT_STREAM
#define NOT_STREAM

#include "3_0_Stream_Const.h"

namespace basic::stream {
    inline const char* getComparisonTarget() {
        return "Null Stream";
    }
    template <class T1, class T2>
    void run_comparison_target(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // NOT_STREAM