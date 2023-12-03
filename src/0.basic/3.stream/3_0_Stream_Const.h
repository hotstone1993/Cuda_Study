#define HOST_INPUT1 0
#define HOST_INPUT2 1
#define DEVICE_INPUT1 2
#define DEVICE_INPUT2 3
#define INPUT_COUNT 4

#define HOST_OUTPUT1 0
#define HOST_OUTPUT2 1
#define DEVICE_OUTPUT1 2
#define DEVICE_OUTPUT2 3
#define OUTPUT_COUNT 4

#define THREADS 1024
#define SIZE (1 << 25)
#define STREAMS 10

typedef int TARGET_INPUT_TYPE;
typedef int TARGET_OUTPUT_TYPE;

#include "utils.cuh"