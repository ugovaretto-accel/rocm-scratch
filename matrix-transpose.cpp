#include <iostream>

#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"

const size_t WIDTH = 1024;
const size_t NUM = (WIDTH * WIDTH);
const int THREADS_PER_BLOCK_X = 4;
const int THREADS_PER_BLOCK_Y = 4;
const int THREADS_PER_BLOCK_Z = 1;

__global__ void
MatrixTranspose(float* out, float* in, size_t width) {
    size_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    size_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    out[y * width + x] = in[x * width + y];

}

void Exec(float* gpuTransposeMatrix, float* gpuMatrix, size_t width) {
    // Lauching kernel from host
    hipLaunchKernelGGL(
        MatrixTranspose,
        dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0,
        gpuTransposeMatrix, gpuMatrix, WIDTH);
}