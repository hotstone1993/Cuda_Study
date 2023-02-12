#ifndef IMAGE_BLUR_SUPPORT
#define IMAGE_BLUR_SUPPORT

#include "1_0_Blur_MT.h"
#include "1_0_Blur.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace image::blur {
    uint8_t *data = nullptr;

    template <class T1, class T2>
    void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        char const* fileName = "../data/1.image/test_image.jpg";
        int width, height, pixel;
        data = stbi_load(fileName, &width, &height, &pixel, 0); // red, green, blue
    }
        
    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        stbi_image_free(data);
    }
}

#endif IMAGE_BLUR_SUPPORT