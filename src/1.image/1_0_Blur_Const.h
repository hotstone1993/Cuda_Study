#define IMAGE_WIDTH 0
#define IMAGE_HEIGHT 1
#define IMAGE_STRIDE 2
#define HOST_INPUT 3
#define DEVICE_INPUT 4
#define INPUT_COUNT 5

#define HOST_OUTPUT 0
#define DEVICE_OUTPUT 1
#define OUTPUT_COUNT 2

#define THREADS 16
#define SIZE 1000

#include "utils.h"

typedef int TARGET_INPUT_TYPE;
typedef uint8_t TARGET_OUTPUT_TYPE;