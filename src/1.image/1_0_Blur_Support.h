#ifndef IMAGE_BLUR_SUPPORT
#define IMAGE_BLUR_SUPPORT

#include "1_0_Blur_MT.h"
#include "1_0_Blur.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace image::blur {
    template <class T1>
    void copyInputs(std::vector<T1*>& inputs) {
        cudaError_t cudaStatus = cudaMemcpy(inputs[DEVICE_INPUT]
                                        , inputs[HOST_INPUT]
                                        , *inputs[IMAGE_WIDTH] * *inputs[IMAGE_HEIGHT] * *inputs[IMAGE_STRIDE] * sizeof(uint8_t)
                                        , cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy failed! (Host to Device)");
        }
    }

    template <class T1, class T2>
    void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        char const* fileName = "../data/1.image/test_image.jpg";
        int width, height, pixel;

        uint8_t *data = stbi_load(fileName, &width, &height, &pixel, 0); // red, green, blue
        if (data == nullptr) {
            throw std::runtime_error("Failed stbi_load\n");
        }

        inputs.resize(INPUT_COUNT);
        inputs[IMAGE_WIDTH] = new T1(width);
        inputs[IMAGE_HEIGHT] = new T1(height);
        inputs[IMAGE_STRIDE] = new T1(pixel);
        size_t inputSize = width * height * pixel / (sizeof(T1) / sizeof(T2));
        inputs[HOST_INPUT] = new T1[inputSize];
        memcpy(inputs[HOST_INPUT], data, sizeof(uint8_t) * width * height * pixel);
        stbi_image_free(data);
        CUDA_MALLOC(inputs[DEVICE_INPUT], inputSize, T1)

        outputs.resize(OUTPUT_COUNT);
        outputs[HOST_OUTPUT] = new T2[width * height * pixel];
        CUDA_MALLOC(outputs[DEVICE_OUTPUT], width * height * pixel, T2)

        copyInputs(inputs);
    }
        
    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        stbi_write_jpg("blured_MT_result.jpg", 
                    *inputs[IMAGE_WIDTH], 
                    *inputs[IMAGE_HEIGHT], 
                    *inputs[IMAGE_STRIDE], 
                    inputs[HOST_INPUT], 100
                );

        delete inputs[IMAGE_WIDTH];
        delete inputs[IMAGE_HEIGHT];
        delete inputs[IMAGE_STRIDE];
        delete[] inputs[HOST_INPUT];
        cudaFree(inputs[DEVICE_INPUT]);
        
        delete[] outputs[HOST_OUTPUT];
        cudaFree(outputs[DEVICE_OUTPUT]);
    }
}

#endif IMAGE_BLUR_SUPPORT