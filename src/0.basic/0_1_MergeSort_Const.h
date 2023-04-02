#define HOST_INPUT 0
#define DEVICE_INPUT 1
#define CPU_INPUT 2
#define DEVICE_RANK_A 3
#define DEVICE_RANK_B 4
#define DEVICE_LIMITS_A 5
#define DEVICE_LIMITS_B 6
#define INPUT_COUNT 7

#define DEVICE_OUTPUT 0
#define OUTPUT_COUNT 1

#define SIZE 1000000
#define THREADS 1024
#define SAMPLE_STRIDE 128
#define DIRECTION true

typedef int TARGET_INPUT_TYPE;
typedef int TARGET_OUTPUT_TYPE;

#include "utils.cuh"