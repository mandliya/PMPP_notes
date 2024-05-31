//
// Description: This code demonstrates vector addition of two vectors of size 2^30 using CPU and GPU.
// The CPU and GPU implementations are compared in terms of time taken.
// For simplicity, there is no error checking in this code.
// The code is written for educational purposes and may not be optimized.
//
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <ctime>

__global__ void vec_add_kernel(float *A, float *B, float *C, unsigned long long n) {
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vec_add_cpu(float *A, float *B, float *C, unsigned long long n) {
    for (unsigned long long i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

void vec_add_gpu(float *A, float *B, float *C, unsigned long long n) {
    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * sizeof(float));
    cudaMalloc((void**)&d_B, n * sizeof(float));
    cudaMalloc((void**)&d_C, n * sizeof(float));
    
    // Copy data from CPU to GPU
    cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    unsigned int num_threads_per_block = 512;
    unsigned int num_blocks = (n + num_threads_per_block - 1) / num_threads_per_block;

    clock_t start = clock();
    vec_add_kernel<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    clock_t end = clock();
    std::cout << "Time taken by GPU kernel: " << (double)(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
    // Copy data from GPU to CPU
    cudaMemcpy(C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    unsigned long long n = (1ULL << 30);
    float *A = (float*)malloc(n * sizeof(float));
    float *B = (float*)malloc(n * sizeof(float));
    float *C_cpu = (float*)malloc(n * sizeof(float));
    float *C_gpu = (float*)malloc(n * sizeof(float));

    for (unsigned long long i = 0; i < n; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    std::cout << "Vector addition of " << n << " elements" << std::endl;

    // Measure time taken by CPU
    clock_t start = clock();
    vec_add_cpu(A, B, C_cpu, n);
    clock_t end = clock();
    std::cout << "Time taken by CPU: " << (double)(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

    // Measure time taken by GPU
    start = clock();
    vec_add_gpu(A, B, C_gpu, n);
    end = clock();
    std::cout << "Time taken by GPU: " << (double)(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "Time taken by GPU includes copying data from CPU to GPU, launching kernel, and copying data from GPU to CPU" << std::endl;


    // Check if the results are correct
    for (unsigned long long i = 0; i < n; i++) {
        if (C_cpu[i] != C_gpu[i]) {
            std::cerr << "Error at index " << i << std::endl;
            break;
        }
    }

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    return 0;
}