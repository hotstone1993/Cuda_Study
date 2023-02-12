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

    __device__ void operator+=(const pixel& other) {
        r += other.r;
        g += other.g;
        b += other.b;
    }
    
    __device__ void operator/=(int value) {
        r /= value;
        g /= value;
        b /= value;
    }
};

__device__ void verticalBlur(TARGET_OUTPUT_TYPE* result, TARGET_OUTPUT_TYPE* image, int x, int y, int w, int h, int intensity) {
    pixelStorage sum;
    int count = 0;

    for (int idx = y - intensity; idx <= y + intensity; ++idx) {
        if (idx < 0 || idx >= h)
            continue;
        ++count;
        sum += reinterpret_cast<pixel*>(image)[idx * w + x];
    }
    sum /= count;

    pixel& target = reinterpret_cast<pixel*>(result)[y * w + x];
    target.r = sum.r;
    target.g = sum.g;
    target.b = sum.b;
}

__device__ void horizontalBlur(TARGET_OUTPUT_TYPE* result, TARGET_OUTPUT_TYPE* image, int x, int y, int w, int h, int intensity) {
    pixelStorage sum;
    int count = 0;

    for (int idx = x - intensity; idx <= x + intensity; ++idx) {
        if (idx < 0 || idx >= w)
            continue;
        ++count;
        sum += reinterpret_cast<pixel*>(image)[y * w + idx];
    }
    sum /= count;

    pixel& target = reinterpret_cast<pixel*>(result)[y * w + x];
    target.r = sum.r;
    target.g = sum.g;
    target.b = sum.b;
}

__global__ void blurImage(TARGET_OUTPUT_TYPE* result, TARGET_OUTPUT_TYPE* image, int w, int h, int pixelStride) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= w || y >= h)
        return;
    verticalBlur(result, image, x, y, w, h, 10);
    horizontalBlur(result, image, x, y, w, h, 10);
}

template <class T1, class T2>
void image::blur::run(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    int blockWidthCount = (*inputs[IMAGE_WIDTH] + THREADS - 1) / THREADS;
    int blockHeightCount = (*inputs[IMAGE_HEIGHT] + THREADS - 1) / THREADS;
    dim3 gridDim(blockWidthCount, blockHeightCount);
    dim3 blockDim(THREADS, THREADS);

    blurImage<<<gridDim, blockDim>>>(outputs[DEVICE_OUTPUT]
                                , reinterpret_cast<T2*>(inputs[DEVICE_INPUT])
                                , *inputs[IMAGE_WIDTH]
                                , *inputs[IMAGE_HEIGHT]
                                , *inputs[IMAGE_STRIDE]);
                                
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("MatMul launch failed\n");
    }
    
    cudaStatus = cudaMemcpy(outputs[HOST_OUTPUT]
                            , outputs[DEVICE_OUTPUT]
                            , (*inputs[IMAGE_WIDTH]) * (*inputs[IMAGE_HEIGHT]) * (*inputs[IMAGE_STRIDE]) * sizeof(T2)
                            , cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed! (Device to Host)");
    }
}

template void image::blur::run(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);