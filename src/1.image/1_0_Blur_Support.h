#ifndef IMAGE_BLUR_SUPPORT
#define IMAGE_BLUR_SUPPORT

#include "1_0_Blur_MT.h"
#include "1_0_Blur.cuh"

namespace image::blur {
    template <class T1, class T2>
    void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {}
        
    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {}
}

#endif IMAGE_BLUR_SUPPORT