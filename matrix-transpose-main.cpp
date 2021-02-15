#include <iostream>
#include <vector>
// hip header file
#include <hip/hip_runtime.h>

using namespace std;

void Exec(float* out, float* in, size_t width);

const size_t WIDTH = 1024;
const size_t NUM = (WIDTH * WIDTH);

// CPU implementation of matrix transpose
void MatrixTransposeCPUReference(float* output, float* input, size_t width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

int main() {
    vector<float> matrix(NUM, 10.f);
    vector<float> transposeMatrix(NUM);
    vector<float> cpuTransposeMatrix(NUM);

    float* gpuMatrix = nullptr;
    float* gpuTransposeMatrix = nullptr;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));
    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, matrix.data(), NUM * sizeof(float), 
              hipMemcpyHostToDevice);
    // Lauching kernel from host
    Exec(gpuTransposeMatrix, gpuMatrix, WIDTH);
    // Memory transfer from device to host
    hipMemcpy(transposeMatrix.data(), gpuTransposeMatrix, matrix.size(), 
              hipMemcpyDeviceToHost);
    // CPU MatrixTranspose computation
    MatrixTransposeCPUReference(cpuTransposeMatrix.data(), matrix.data(), 
                                matrix.size());
    // verify the results
    int errors = 0;
    double eps = 1.0E-6;
    for (size_t i = 0; i != NUM; i++) {
        if (std::abs(transposeMatrix[i] - cpuTransposeMatrix[i]) > eps) {
            errors++;
        }
    }
    if (errors != 0) {
        printf("FAILED: %d errors\n", errors);
    } else {
        printf("PASSED!\n");
    }
    // free the resources on device side
    hipFree(gpuMatrix);
    hipFree(gpuTransposeMatrix);
    return errors;
}