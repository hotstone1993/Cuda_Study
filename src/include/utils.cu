#include "utils.cuh"

__device__ uint32_t __mysmid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ uint32_t __mywarpid() {
    uint32_t warpid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
    return warpid;
}
  
__device__ uint32_t __mylaneid() {
    uint32_t laneid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}