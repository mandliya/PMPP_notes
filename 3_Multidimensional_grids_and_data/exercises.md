# Exercises

## 3.1
In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.
    a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.
    b. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.
    c. Analyze the pros and cons of each of the two kernel designs.

Solution:
Let's write a row-wise matrix multiplication kernel first. We will assign each thread to a row of the output matrix. The kernel will look like this:

```cpp
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
```

Similarly, let's write a column-wise matrix multiplication kernel. We will assign each thread to a column of the output matrix. The kernel will look like this:

```cpp
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
```
Complete code lives at [exercise_1](code/exercise_1.cu)
On my machine with NVIDIA Tesla T4, running these two GPU kernels along with CPU code, this is the result I see:

Matrix A is of size 800x1000 and Matrix B is of size 1000x800,

```bash
CPU time: 3.64059 s
GPU row-wise time: 3.47706 s
GPU col-wise time: 2.54797 s
```

The row-wise kernel is slower than the column-wise kernel. Probably because column-wise kernel has better memory access patterns than the row-wise kernel. In row-wise kernel, for one output row, we are accessing a row of matrix A and all columns of matrix B. In column-wise kernel, for one output column, we are accessing all rows of matrix A and a column of matrix B. In future chapters we will see why one is better that another. My guess is memory coalescing.

To me, row wise kernel is easier to understand and write. But, column wise kernel is faster. 

## 3.2
A matrix-vector multiplication takes an input matrix B and a vector C and
produces one output vector A. Each element of the output vector A is the dot
product of one row of the input matrix B and C, that is, $A[i] = \sum^jB[i][j] + C[j]$.
For simplicity we will handle only square matrices whose elements are single-
precision floating-point numbers. Write a matrix-vector multiplication kernel and
the host stub function that can be called with four parameters: pointer to the output
matrix, pointer to the input matrix, pointer to the input vector, and the number of
elements in each dimension. Use one thread to calculate an output vector element.

Solution:
```cpp
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
```
Complete code lives in [exercise_2](code/exercise_2.cu)

## 3.3 
Consider the following CUDA kernel and the corresponding host function that calls it:

```cpp
01 __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
02     unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
03     unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
04     if(row < M && col < N) {
05         b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
06     }
07 }

08 void foo(float* a_d, float* b_d) {
09     unsigned int M = 150;
10     unsigned int N = 300;
11     dim3 bd(16, 32);
12     dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);
13     foo_kernel<<< gd, bd >>>(a_d, b_d, M, N);
14 }
```

a. What is the number of threads per block?
b. What is the number of threads in the grid?
c. What is the number of blocks in the grid?
d. What is the number of threads that execute the code on line 05?

Solution:
a. The number of threads per block is 16 x 32 = 512.
b. The number of threads in the grid is
    - Blocks:  (N - 1)/16 + 1 x (M - 1)/32 + 1 = 20 x 6 = 120 (applying ceiling on division)
    - Threads: 120 x 512 = 61440
c. The number of blocks in the grid is 20 x 6 = 120.
d. The number of threads that execute the code on line 05 is 150 x 300 = 45000.

## 3.4
Consider a 2D matrix with a width of 400 and a height of 500. The matrix is
stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:
a. If the matrix is stored in row-major order.
b. If the matrix is stored in column-major order

Solution:
a. If the matrix is stored in row-major order, the array index of the matrix element at row 20 and column 10 is 20 * 400 + 10 = 8010.
b. If the matrix is stored in column-major order, the array index of the matrix element at row 20 and column 10 is 10 * 500 + 20 = 5020.

## 3.5 
Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5.

Solution:
For row major order, let's think this through step by step.
We will have 3 indices to work with.
- let's think this way, imagine our data is laid out like this example of 2 x 3 x 3
```python
[
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ]
]
```
This will be laid out in memory like this.
```
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
```
Say we want to find index at (1, 2, 2), we can calculate it like this:
- First index will index the 2nd 2D array, so 1 * 3 * 3 = 9
- Second index will index the 3rd row of the 2nd 2D array, so 2 * 3 = 6
- Third index will index the 3rd element of the 3rd row of the 2nd 2D array, so 2
- Thus the index will be 9 + 6 + 2 = 17 (which stores value 18, 0 based indexing)

Similarly, lets work out the index for (10, 20, 5) in a 400 x 500 x 300 tensor.
- First index will be 11th 2D array, so 10 * 500 * 300 = 1,500,000
- Second index will be 21st row of the 11th 2D array, so 20 * 300 = 6,000
- Third index will be 6th element of the 21st row of the 10th 2D array, so 5
- Thus the index will be 1,500,000 + 6,000 + 5 = 1,506,005

