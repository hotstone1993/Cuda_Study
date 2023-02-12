#ifndef IMAGE_BLUR_MT
#define IMAGE_BLUR_MT

#include "1_0_Blur_Const.h"

namespace image::blur {
    template <class T1, class T2>
    void run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs);
}

#endif // IMAGE_BLUR_MT