#include "1_0_Blur.cuh"

cudaTextureObject_t image::blur::rgbaTex = 0;
cudaArray *image::blur::textureArray = nullptr;

__global__ void verticalBlurImage(TARGET_OUTPUT_TYPE* result, TARGET_OUTPUT_TYPE* image, int w, int h, int intensity) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    pixel* input = reinterpret_cast<pixel*>(image);
    pixel* output = reinterpret_cast<pixel*>(result);

    if (x >= w)
        return;

    float scale = (float)((intensity << 1) + 1);
    pixelStorage sum = input[x];
    sum *= intensity;

    for (size_t y = 0; y < intensity + 1; ++y) {
        sum += input[y * w + x];
    }
    setPixel(output[x], sum / scale);

    for (size_t y = 1; y < intensity + 1; ++y) {
        sum -= input[x];
        sum += input[(y + intensity) * w + x];
        setPixel(output[y * w + x], sum / scale);
    }

    for (size_t y = intensity + 1; y < h - intensity; ++y) {
        sum -= input[(y - intensity - 1) * w + x];
        sum += input[(y + intensity) * w + x];
        setPixel(output[y * w + x], sum / scale);
    }

    for (size_t y = h - intensity; y < h; ++y) {
        sum += input[(h - 1) * w + x];
        sum -= input[(y - intensity - 1) * w + x];
        setPixel(output[y * w + x], sum / scale);
    }
}

__global__ void horizontalBlurImage(TARGET_OUTPUT_TYPE* result, cudaTextureObject_t image, int w, int h, int intensity) {
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    pixel* output = reinterpret_cast<pixel*>(result);

    if (y >= h)
        return;

    float scale = (float)((intensity << 1) + 1);
    pixelStorage sum{};

    for (int x = -intensity; x <= intensity; ++x) {
        sum += tex2D<uchar4>(image, x, y);
    }
    setPixel(output[y * w], sum / scale);
    
    for (int x = 1; x < w; ++x) {
        sum -= tex2D<uchar4>(image, x - intensity - 1, y);
        sum += tex2D<uchar4>(image, x + intensity, y);
        setPixel(output[y * w + x], sum / scale);
    }
}

template <class T1, class T2>
void image::blur::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    int width = *inputs[IMAGE_WIDTH];
    int height = *inputs[IMAGE_HEIGHT];
    int pixel = *inputs[IMAGE_STRIDE];
    TARGET_OUTPUT_TYPE* buffer1 = reinterpret_cast<T2*>(inputs[DEVICE_INPUT]);
    TARGET_OUTPUT_TYPE* buffer2 = outputs[DEVICE_OUTPUT];
    cudaArray *textureArray;

    horizontalBlurImage<<<height / (THREADS - 1), THREADS>>>(buffer1
                                , rgbaTex
                                , width
                                , height
                                , intensity);
                                
    checkCudaError(cudaGetLastError(), "vertical blur failed - ");

    verticalBlurImage<<<width / (THREADS - 1), THREADS>>>(buffer2
                                , buffer1
                                , width
                                , height
                                , intensity);
                                
    checkCudaError(cudaGetLastError(), "horizontal blur failed - ");
    
    checkCudaError(cudaMemcpy(outputs[HOST_OUTPUT_CUDA]
                            , outputs[DEVICE_OUTPUT]
                            , width * height * pixel * sizeof(T2)
                            , cudaMemcpyDeviceToHost),
                            "cudaMemcpy failed! (Device to Host) - ");
}

template void image::blur::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);