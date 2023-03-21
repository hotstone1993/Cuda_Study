#define HOST_INPUT 0
#define DEVICE_INPUT 1
#define CPU_INPUT 2
#define INPUT_COUNT 3

#define DEVICE_OUTPUT 0
#define OUTPUT_COUNT 1

#define SIZE 1000000
#define THREADS 1024
#define DIRECTION true

typedef int TARGET_INPUT_TYPE;
typedef int TARGET_OUTPUT_TYPE;

#include "utils.cuh"