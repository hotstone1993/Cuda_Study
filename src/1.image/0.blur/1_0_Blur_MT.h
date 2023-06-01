#ifndef IMAGE_BLUR_MT
#define IMAGE_BLUR_MT

#include "1_0_Blur_Const.h"
#include "concept_utils.h"

namespace image::blur {
    inline const char* getComparisonTarget() {
        return "Multi Thread";
    }
    template <class T1, class T2>
    void run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // IMAGE_BLUR_MT