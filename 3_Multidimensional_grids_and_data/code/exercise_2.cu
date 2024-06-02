/**
 * A matrix-vector multiplication takes an input matrix B and a vector C and
produces one output vector A. Each element of the output vector A is the dot
product of one row of the input matrix B and C, that is, $A[i] = \sum^jB[i][j] + C[j]$.
For simplicity we will handle only square matrices whose elements are single-
precision floating-point numbers. Write a matrix-vector multiplication kernel and
the host stub function that can be called with four parameters: pointer to the output
matrix, pointer to the input matrix, pointer to the input vector, and the number of
elements in each dimension. Use one thread to calculate an output vector element.

No error handling for simplicity.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void mat_vec_mul_kernel(int *A, int *B, int* C, unsigned int n) {
    // A is output vector of size n
    // B is input matrix of size n x n
    // C is input vector of size n
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        int sum = 0;
        for (unsigned int i = 0; i < n; i++) {
            sum += (B[row * n + i] * C[i]);
        }
        A[row] = sum;
    }
}

void mat_vec_mul_gpu(int *A, int *B, int *C, unsigned int n) {
    int *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, n * sizeof(int));
    cudaMalloc((void**)&C_d, n * sizeof(int));
    cudaMalloc((void**)&B_d, n * n * sizeof(int));

    cudaMemcpy(B_d, B, sizeof(int)* n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(int)*n, cudaMemcpyHostToDevice);

    unsigned int numThreadsPerBlock = 32;
    unsigned int numBlocks = (n + 32 - 1)/32;
    mat_vec_mul_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, n);

    cudaMemcpy(A, A_d, sizeof(int)*n, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void mat_vec_mul_cpu(int* A, int *B, int *C, int n) {
    for (unsigned int i = 0; i < n; i++) {
        A[i] = 0;
        for (unsigned int j = 0; j < n; j++) {
            A[i] += (B[i * n + j] * C[j]);
        }
    }
}


int main() {
    int n = 1000;
    int* A_cpu = new int[n];
    int* A_gpu = new int[n];
    int* B = new int[n * n];
    int* C = new int[n];

    for (int i = 0; i < n; i++) {
        C[i] = rand() % 100;
    }

    for (int i = 0; i < n * n; i++) {
        B[i] = rand() % 100;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    mat_vec_mul_cpu(A_cpu, B, C, n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
    std::cout << "CPU time: " << elapsed_cpu.count() << " s" << std::endl;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    mat_vec_mul_gpu(A_gpu, B, C, n);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gpu = end_gpu - start_gpu;
    std::cout << "GPU time: " << elapsed_gpu.count() << " s" << std::endl;

    for (int i = 0; i < n; i++) {
        if (A_cpu[i] != A_gpu[i]) {
            std::cerr << "Mismatch at index " << i << std::endl;
            break;
        }
    }

    delete[] A_cpu;
    delete[] A_gpu;
    delete[] B;
    delete[] C;
}