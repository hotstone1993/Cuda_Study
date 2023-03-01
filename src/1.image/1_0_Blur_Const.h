#define IMAGE_WIDTH 0
#define IMAGE_HEIGHT 1
#define IMAGE_STRIDE 2
#define HOST_INPUT 3
#define DEVICE_INPUT 4
#define INPUT_COUNT 5

#define HOST_OUTPUT 0
#define DEVICE_OUTPUT 1
#define OUTPUT_COUNT 2

#define THREADS 64

#include "utils.h"

typedef int TARGET_INPUT_TYPE;
typedef uint8_t TARGET_OUTPUT_TYPE;

#ifndef IMAGE_BLUR_CONST
#define IMAGE_BLUR_CONST

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

#endif // IMAGE_BLUR_CONST