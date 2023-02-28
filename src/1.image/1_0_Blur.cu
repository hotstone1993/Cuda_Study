#include "1_0_Blur.cuh"

cudaTextureObject_t image::blur::rgbaTex = 0;
cudaArray *image::blur::textureArray = nullptr;

struct pixel {
    TARGET_OUTPUT_TYPE r = 0;
    TARGET_OUTPUT_TYPE g = 0;
    TARGET_OUTPUT_TYPE b = 0;
    TARGET_OUTPUT_TYPE a = 0;
};

struct pixelStorage {
    uint32_t r = 0;
    uint32_t g = 0;
    uint32_t b = 0;
    uint32_t a = 0;

    __device__ pixelStorage() {}
    __device__ pixelStorage(const pixel& other): r(other.r), g(other.g), b(other.b), a(other.a) {}
    __device__ pixelStorage(const pixelStorage& other): r(other.r), g(other.g), b(other.b), a(other.a) {}

    __device__ void operator+=(const pixel& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        a += other.a;
    }
    __device__ void operator+=(const uchar4& other) {
        r += other.x;
        g += other.y;
        b += other.z;
        a += other.w;
    }

    __device__ void operator-=(const pixel& other) {
        r -= other.r;
        g -= other.g;
        b -= other.b;
        a -= other.a;
    }
    __device__ void operator-=(const uchar4& other) {
        r -= other.x;
        g -= other.y;
        b -= other.z;
        a -= other.w;
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
        a /= value;
    }
    
    __device__ void operator*=(int value) {
        r *= value;
        g *= value;
        b *= value;
        a *= value;
    }
};

__device__ inline void setPixel(pixel& target, pixelStorage result) {
    target.r = result.r;
    target.g = result.g;
    target.b = result.b;
    target.a = result.a;
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
    int intensity = 50;
    TARGET_OUTPUT_TYPE* buffer1 = reinterpret_cast<T2*>(inputs[DEVICE_INPUT]);
    TARGET_OUTPUT_TYPE* buffer2 = outputs[DEVICE_OUTPUT];
    cudaArray *textureArray;

    horizontalBlurImage<<<height / (THREADS - 1), THREADS>>>(buffer1
                                , rgbaTex
                                , width
                                , height
                                , intensity);
                                
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::string error = "vertical blur failed - ";
        error += cudaGetErrorString(cudaStatus);
        throw std::runtime_error(error);
    }

    verticalBlurImage<<<width / (THREADS - 1), THREADS>>>(buffer2
                                , buffer1
                                , width
                                , height
                                , intensity);
                                
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::string error = "horizontal blur failed - ";
        error += cudaGetErrorString(cudaStatus);
        throw std::runtime_error(error);
    }
    
    cudaStatus = cudaMemcpy(outputs[HOST_OUTPUT]
                            , outputs[DEVICE_OUTPUT]
                            , width * height * pixel * sizeof(T2)
                            , cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::string error = "cudaMemcpy failed! (Device to Host) - ";
        error += cudaGetErrorString(cudaStatus);
        throw error;
    }
}

template void image::blur::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);