#include<iostream>
#include<cuda_runtime.h>
#include<time.h>

void mat_mul_cpu(int *A, int* B, int* C, unsigned int m, unsigned int n, unsigned int p) {
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (unsigned int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

__global__ void mat_mul_kernel(int *A, int* B, int* C, unsigned int m, unsigned int n, unsigned int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        C[row * p + col] = 0;
        for (unsigned int i = 0; i < n; i++) {
            C[row * p + col] += A[row * n + i] * B[i * p + col];
        }
    }
}


void mat_mul_gpu(int *A, int* B, int* C, unsigned int m, unsigned int n, unsigned int p) {
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(int));
    cudaMalloc(&d_B, n * p * sizeof(int));
    cudaMalloc(&d_C, m * p * sizeof(int));

    cudaMemcpy(d_A, A, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((p + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    mat_mul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, p);

    cudaMemcpy(C, d_C, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    unsigned int M = 400;
    unsigned int N = 500;
    unsigned int P = 300;

    int *A = new int[M * N];
    int *B = new int[N * P];
    int *C_cpu = new int[M * P];
    int *C_gpu = new int[M * P];

    for (unsigned int i = 0; i < M * N; i++) {
        A[i] = rand() % 100;
    }

    for (unsigned int i = 0; i < N * P; i++) {
        B[i] = rand() % 100;
    }

    clock_t start, end;
    double cpu_time_used, gpu_time_used;
    start = clock();
    mat_mul_cpu(A, B, C_cpu, M, N, P);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    start = clock();
    mat_mul_gpu(A, B, C_gpu, M, N, P);
    end = clock();
    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    std::cout << "CPU time: " << cpu_time_used << std::endl;
    std::cout << "GPU time: " << gpu_time_used << std::endl;

    // validate both results
    for (unsigned int i = 0; i < M * P; i++) {
        if (C_cpu[i] != C_gpu[i]) {
            std::cerr << "Error: CPU and GPU results do not match" << std::endl;
            break;
        }
    }
    std::cout << "Results match!" << std::endl;
    
    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;

}