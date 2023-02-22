#include "1_0_Blur_MT.h"

constexpr size_t THREAD_COUNT = 8;
constexpr int intensity = 500;

struct pixel {
    TARGET_OUTPUT_TYPE r = 0;
    TARGET_OUTPUT_TYPE g = 0;
    TARGET_OUTPUT_TYPE b = 0;
};

struct pixelStorage {
    uint32_t r = 0;
    uint32_t g = 0;
    uint32_t b = 0;

    pixelStorage(const pixel& other): r(other.r), g(other.g), b(other.b) {}
    pixelStorage(const pixelStorage& other): r(other.r), g(other.g), b(other.b) {}

    void operator+=(const pixel& other) {
        r += other.r;
        g += other.g;
        b += other.b;
    }

    void operator-=(const pixel& other) {
        r -= other.r;
        g -= other.g;
        b -= other.b;
    }
    
    pixelStorage operator/(int value) {
        pixelStorage newStorage(*this);
        newStorage /= value;

        return newStorage;
    }
    
    void operator/=(int value) {
        r /= value;
        g /= value;
        b /= value;
    }
    
    void operator*=(int value) {
        r *= value;
        g *= value;
        b *= value;
    }
};

inline void setPixel(pixel& target, pixelStorage result) {
    target.r = result.r;
    target.g = result.g;
    target.b = result.b;
}

void verticalBlur(TARGET_OUTPUT_TYPE* image, TARGET_OUTPUT_TYPE* result, int w, int h, int x) {
    pixel* input = reinterpret_cast<pixel*>(image);
    pixel* output = reinterpret_cast<pixel*>(result);

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
void horizontalBlur(TARGET_OUTPUT_TYPE* image, TARGET_OUTPUT_TYPE* result, int w, int h, int y) {
    pixel* input = reinterpret_cast<pixel*>(image);
    pixel* output = reinterpret_cast<pixel*>(result);

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
void image::blur::run_mt(std::vector<T1*>& inputs, std::vector<T2*>& outputs) {
    std::vector<std::future<void>> futures;
    ThreadPool tp(THREAD_COUNT);

    int width = *inputs[IMAGE_WIDTH];
    int height = *inputs[IMAGE_HEIGHT];
    TARGET_OUTPUT_TYPE* buffer1 = reinterpret_cast<T2*>(inputs[HOST_INPUT]);
    TARGET_OUTPUT_TYPE* buffer2 = outputs[HOST_OUTPUT];

    for (size_t x = 0; x < width; ++x) {
        futures.emplace_back(tp.addJob(verticalBlur, buffer1, buffer2, width, height, x));
    }
    
    for (auto& f : futures) {
        f.get(); // for exception
    }
    futures.clear();
    
    for (size_t y = 0; y < height; ++y) {
        futures.emplace_back(tp.addJob(horizontalBlur, buffer2, buffer1, width, height, y));
    }
    
    for (auto& f : futures) {
        f.get(); // for exception
    }
}

template void image::blur::run_mt(std::vector<TARGET_INPUT_TYPE*>& inputs, std::vector<TARGET_OUTPUT_TYPE*>& outputs);