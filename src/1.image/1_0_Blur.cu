#include "1_0_Blur.cuh"

struct pixel {
    TARGET_OUTPUT_TYPE r = 0;
    TARGET_OUTPUT_TYPE g = 0;
    TARGET_OUTPUT_TYPE b = 0;
};

struct pixelStorage {
    uint32_t r = 0;
    uint32_t g = 0;
    uint32_t b = 0;

    __device__ pixelStorage(const pixel& other): r(other.r), g(other.g), b(other.b) {}
    __device__ pixelStorage(const pixelStorage& other): r(other.r), g(other.g), b(other.b) {}

    __device__ void operator+=(const pixel& other) {
        r += other.r;
        g += other.g;
        b += other.b;
    }

    __device__ void operator-=(const pixel& other) {
        r -= other.r;
        g -= other.g;
        b -= other.b;
    }
    
    __device__ pixelStorage operator/(int value) {
        pixelStorage newStorage(*this);
        newStorage /= value;

        return newStorage;
    }
    
    __device__ void operator/=(int value) {
        r /= value;
        g /= value;
        b /= value;
    }
    
    __device__ void operator*=(int value) {
        r *= value;
        g *= value;
        b *= value;
    }
};

__device__ inline void setPixel(pixel& target, pixelStorage result) {
    target.r = result.r;
    target.g = result.g;
    target.b = result.b;
}

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

__global__ void horizontalBlurImage(TARGET_OUTPUT_TYPE* result, TARGET_OUTPUT_TYPE* image, int w, int h, int intensity) {
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    pixel* input = reinterpret_cast<pixel*>(image);
    pixel* output = reinterpret_cast<pixel*>(result);

    if (y >= h)
        return;

    float scale = (float)((intensity << 1) + 1);
    pixelStorage sum = input[y * w];
    sum *= intensity;

    for (size_t x = 0; x < intensity + 1; ++x) {
        sum += input[y * w + x];
    }
    setPixel(output[y * w], sum / scale);

    for (size_t x = 1; x < intensity + 1; ++x) {
        sum -= input[y * w];
        sum += input[y * w + x + intensity];
        setPixel(output[y * w + x], sum / scale);
    }

    for (size_t x = intensity + 1; x < w - intensity; ++x) {
        sum -= input[y * w + x - intensity - 1];
        sum += input[y * w + x + intensity];
        setPixel(output[y * w + x], sum / scale);
    }

    for (size_t x = w - intensity; x < w; ++x) {
        sum -= input[y * w + x - intensity - 1];
        sum += input[(y + 1) * w - 1];
        setPixel(output[y * w + x], sum / scale);
    }
}

template <class T1, class T2>
void image::blur::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    int intensity = 500;
    int width = *inputs[IMAGE_WIDTH];
    int height = *inputs[IMAGE_HEIGHT];
    TARGET_OUTPUT_TYPE* buffer1 = reinterpret_cast<T2*>(inputs[DEVICE_INPUT]);
    TARGET_OUTPUT_TYPE* buffer2 = outputs[DEVICE_OUTPUT];

    verticalBlurImage<<<width / (THREADS - 1), THREADS>>>(buffer2
                                , buffer1
                                , width
                                , height
                                , intensity);
                                
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("MatMul launch failed\n");
    }

    horizontalBlurImage<<<height / (THREADS - 1), THREADS>>>(buffer1
                                , buffer2
                                , width
                                , height
                                , intensity);
                                
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("MatMul launch failed\n");
    }
    
    cudaStatus = cudaMemcpy(outputs[HOST_OUTPUT]
                            , buffer1
                            , width * height * (*inputs[IMAGE_STRIDE]) * sizeof(T2)
                            , cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed! (Device to Host)");
    }
}

template void image::blur::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);