#define HOST_INPUT1 0
#define DEVICE_INPUT1 1
#define INPUT_COUNT 2

#define HOST_OUTPUT1 0
#define DEVICE_OUTPUT1 1
#define OUTPUT_COUNT 2

#define THREADS 1024
#define SIZE 1 << 20

typedef int TARGET_INPUT_TYPE;
typedef int TARGET_OUTPUT_TYPE;

#include "utils.cuh"