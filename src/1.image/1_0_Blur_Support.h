#ifndef IMAGE_BLUR_SUPPORT
#define IMAGE_BLUR_SUPPORT

#include "1_0_Blur_MT.h"
#include "1_0_Blur.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace image::blur {
    
    void initTexture(int width, int height, void *pImage) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

        size_t bytesPerElem = sizeof(uchar4);

        checkCudaError(cudaMallocArray(&textureArray, &channelDesc, width, height), "cudaMallocArray failed! - ");

        checkCudaError(cudaMemcpy2DToArray(
            textureArray, 0, 0, pImage, width * bytesPerElem, width * bytesPerElem, height,
            cudaMemcpyHostToDevice), "cudaMemcpy2DToArray failed! - ");

        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = textureArray;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = false;
        texDescr.filterMode = cudaFilterModePoint;
        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.addressMode[1] = cudaAddressModeWrap;
        texDescr.readMode = cudaReadModeElementType;

        checkCudaError(cudaCreateTextureObject(&rgbaTex, &texRes, &texDescr, NULL), "cudaCreateTextureObject failed! - ");
    }
    
    template <class T1>
    void copyInputs(std::vector<T1*>& inputs) {
        checkCudaError(cudaMemcpy(inputs[DEVICE_INPUT]
                                        , inputs[HOST_INPUT1]
                                        , *inputs[IMAGE_WIDTH] * *inputs[IMAGE_HEIGHT] * *inputs[IMAGE_STRIDE] * sizeof(uint8_t)
                                        , cudaMemcpyHostToDevice),
                                        "cudaMemcpy failed! (Host to Device) - ");
    }

    template <class T1, class T2>
    requires (std::integral<T1> && sizeof(T1) >= 4)
    void setup(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        char const* fileName = "../data/1.image/Sample-png-image-30mb.png";
        int width, height, pixel;

        uint8_t *data = stbi_load(fileName, &width, &height, &pixel, 0); // red, green, blue, alpha
        if (data == nullptr) {
            throw std::runtime_error("Failed stbi_load\n");
        }

        inputs.resize(INPUT_COUNT);
        inputs[IMAGE_WIDTH] = new T1(width);
        inputs[IMAGE_HEIGHT] = new T1(height);
        inputs[IMAGE_STRIDE] = new T1(pixel);
        size_t inputSize = width * height * pixel * sizeof(uint8_t);
        inputs[HOST_INPUT1] = new T1[inputSize];
        inputs[HOST_INPUT2] = new T1[inputSize];
        memcpy(inputs[HOST_INPUT1], data, inputSize);
        initTexture(width, height, data);
        stbi_image_free(data);
        CUDA_MALLOC(inputs[DEVICE_INPUT], inputSize, T1)

        outputs.resize(OUTPUT_COUNT);
        outputs[HOST_OUTPUT_MT] = new T2[width * height * pixel];
        outputs[HOST_OUTPUT_CUDA] = new T2[width * height * pixel];
        CUDA_MALLOC(outputs[DEVICE_OUTPUT], width * height * pixel, T2)

        copyInputs(inputs);
    }
        
    template <class T1, class T2>
    void destroy(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
        stbi_write_jpg("blured_result_mt.jpg", 
                    *inputs[IMAGE_WIDTH], 
                    *inputs[IMAGE_HEIGHT], 
                    *inputs[IMAGE_STRIDE], 
                    reinterpret_cast<T2*>(inputs[HOST_INPUT1]), 100
                );

        stbi_write_jpg("blured_result_cuda.jpg", 
                    *inputs[IMAGE_WIDTH], 
                    *inputs[IMAGE_HEIGHT], 
                    *inputs[IMAGE_STRIDE], 
                    outputs[HOST_OUTPUT_CUDA], 100
                );

        delete inputs[IMAGE_WIDTH];
        delete inputs[IMAGE_HEIGHT];
        delete inputs[IMAGE_STRIDE];
        delete[] inputs[HOST_INPUT1];
        delete[] inputs[HOST_INPUT2];
        cudaFree(inputs[DEVICE_INPUT]);
        
        delete[] outputs[HOST_OUTPUT_MT];
        delete[] outputs[HOST_OUTPUT_CUDA];
        cudaFree(outputs[DEVICE_OUTPUT]);
        cudaFreeArray(textureArray);
        cudaDestroyTextureObject(rgbaTex);
    }
}

#endif IMAGE_BLUR_SUPPORT