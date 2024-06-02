/**
 * A very simple matrix multiplication example to demonstrate the difference between row-wise and column-wise matrix multiplication.
 * It doesn't handle error checking and is not optimized for performance.
*/
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void row_wise_mat_mul_kernel(int *A, int* B, int* C, unsigned int m, unsigned int n, unsigned int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < m) {
        for (unsigned int j = 0; j < p; j++) {
            C[row * p + j] = 0;
            for (unsigned int k = 0; k < n; k++) {
                C[row * p + j] += A[row * n + k] * B[k * p + j];
            }
        }
    }
}

__global__ void col_wise_mat_mul_kernel(int *A, int *B, int *C, unsigned int m, unsigned int n, unsigned int p) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < p) {
        for (unsigned int i = 0; i < m; i++) {
            C[i * p + col] = 0;
            for (unsigned int k = 0; k < n; k++) {
                C[i * p + col] += A[i * n + k] * B[k * p + col];
            }
        }
    }
}

void row_wise_mat_mul_gpu(int *A, int* B, int* C, unsigned int m, unsigned int n, unsigned int p) {
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(int));
    cudaMalloc(&d_B, n * p * sizeof(int));
    cudaMalloc(&d_C, m * p * sizeof(int));

    cudaMemcpy(d_A, A, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((p + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    row_wise_mat_mul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, p);

    cudaMemcpy(C, d_C, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void col_wise_mat_mul_gpu(int *A, int* B, int* C, unsigned int m, unsigned int n, unsigned int p) {
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(int));
    cudaMalloc(&d_B, n * p * sizeof(int));
    cudaMalloc(&d_C, m * p * sizeof(int));

    cudaMemcpy(d_A, A, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((p + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    col_wise_mat_mul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, p);

    cudaMemcpy(C, d_C, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matmul_cpu(int *A, int *B, int *C, unsigned int m, unsigned int n, unsigned int p) {
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (unsigned int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

int main() {
    unsigned int M = 800;
    unsigned int N = 1000;
    unsigned int P = 800;

    int *A = new int[M * N];
    int *B = new int[N * P];
    int *C_cpu = new int[M * P];
    int *C_gpu_row = new int[M * P];
    int *C_gpu_col = new int[M * P];

    for (unsigned int i = 0; i < M * N; i++) {
        A[i] = rand() % 100;
    }

    for (unsigned int i = 0; i < N * P; i++) {
        B[i] = rand() % 100;
    }

    // Measure time for CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C_cpu, M, N, P);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
    std::cout << "CPU time: " << elapsed_cpu.count() << " s" << std::endl;

    auto start_gpu_row = std::chrono::high_resolution_clock::now();
    row_wise_mat_mul_gpu(A, B, C_gpu_row, M, N, P);
    auto end_gpu_row = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gpu_row = end_gpu_row - start_gpu_row;
    std::cout << "GPU row-wise time: " << elapsed_gpu_row.count() << " s" << std::endl;

    for (unsigned int i = 0; i < M * P; i++) {
        if (C_cpu[i] != C_gpu_row[i]) {
            std::cout << "Mismatch at index " << i << " expected " << C_cpu[i] << " got " << C_gpu_row[i] << std::endl;
            break;
        }
    }

    auto start_gpu_col = std::chrono::high_resolution_clock::now();
    col_wise_mat_mul_gpu(A, B, C_gpu_col, M, N, P);
    auto end_gpu_col = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_gpu_col = end_gpu_col - start_gpu_col;
    std::cout << "GPU col-wise time: " << elapsed_gpu_col.count() << " s" << std::endl;

    for (unsigned int i = 0; i < M * P; i++) {
        if (C_cpu[i] != C_gpu_col[i]) {
            std::cout << "Mismatch at index " << i << " expected " << C_cpu[i] << " got " << C_gpu_col[i] << std::endl;
            break;
        }
    }

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu_row;
    delete[] C_gpu_col;

    return 0;
}