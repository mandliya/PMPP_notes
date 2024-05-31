# Chapter 2 Exercises

## Exercise 2.1
If we want to use each thread in a grid to calculate one output element of a
vector addition, what would be the expression for mapping the thread/block
indices to the data index (i)?

- (A) i=threadIdx.x + threadIdx.y;
- (B) i=blockIdx.x + threadIdx.x;
- (C) i=blockIdx.x * blockDim.x + threadIdx.x;
- (D) i=blockIdx.x * threadIdx.x;

Solution: (C) i=blockIdx.x * blockDim.x + threadIdx.x;

As blockIdx.x is the block index and blockDim.x is the number of threads in a block, the expression $i=blockIdx.x * blockDim.x + threadIdx.x$ will map the thread/block indices to the data index.

## Exercise 2.2
Assume that we want to use each thread to calculate two adjacent elements of
a vector addition. What would be the expression for mapping the thread/block
indices to the data index (i) of the first element to be processed by a thread?
- (A) i=blockIdx.x*blockDim.x + threadIdx.x +2;
- (B) i=blockIdx.x*threadIdx.x*2;
- (C) i=(blockIdx.x*blockDim.x + threadIdx.x)*2;
- (D) i=blockIdx.x*blockDim.x*2 + threadIdx.x;

Solution: (C) i=(blockIdx.x*blockDim.x + threadIdx.x)*2;

We want to use each thread to calculate two adjacent elements of a vector addition. So, the expression $i=(blockIdx.x*blockDim.x + threadIdx.x)*2$ will map the thread/block indices to the data index of the first element to be processed by a thread. For example: thread 0 will process elements 0 and 1, thread 1 will process elements 2 and 3, and so on.

## Exercise 2.3
We want to use each thread to calculate two elements of a vector addition.
Each thread block processes 2*blockDim.x consecutive elements that form
two sections. All threads in each block will process a section first, each
processing one element. They will then all move to the next section, each
processing one element. Assume that variable i should be the index for the
first element to be processed by a thread. What would be the expression for
mapping the thread/block indices to data index of the first element?

- (A) i=blockIdx.x*blockDim.x + threadIdx.x + 2;
- (B) i=blockIdx.x*threadIdx.x*2;
- (C) i=(blockIdx.x*blockDim.x + threadIdx.x)*2;
- (D) i=blockIdx.x*blockDim.x*2 + threadIdx.x;

Solution: (D) i=blockIdx.x*blockDim.x*2 + threadIdx.x;

## Exercise 2.4
For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

- (A) 8000
- (B) 8196
- (C) 8192
- (D) 8200

Solution: (C) 8192
So first we get the number of blocks needed to cover all output elements: 8000/1024 = 7.8125. Since we need a minimum number of blocks, we need to round up to 8 blocks. The number of threads in the grid will be 8 * 1024 = 8192.

## Exercise 2.5
If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?

- (A) n
- (B) v
- (C) n * sizeof(int)
- (D) v * sizeof(int)

Solution: (D) v * sizeof(int)
We want to allocate an array of v integer elements in the CUDA device global memory. The second argument of the `cudaMalloc` call should be the size of the memory to allocate in bytes. Since each integer element requires `sizeof(int)` bytes, the appropriate expression for the second argument of the `cudaMalloc` call would be `v * sizeof(int)`.

## Exercise 2.6
If we want to allocate an array of `n` floating-point elements and have a floating-point pointer variable `A_d` to point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc()` call?
- (A) `n`
- (B) `(void *) A_d`
- (C) `*A_d`
- (D) `(void **) &A_d`

Solution: (B) `(void *) A_d`
To allocate memory, the first argument of the `cudaMalloc()` is an address of a pointer to the allocated memory. Since we have a floating-point pointer variable `A_d` to point to the allocated memory, the appropriate expression for the first argument of the `cudaMalloc()` call would be `(void *) A_d`.

## Exercise 2.7
If we want to copy 3000 bytes of data from host array `A_h` (A_h is a pointer to element 0 of the source array) to device array `A_d` (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?

- (A) `cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);`
- (B) `cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);`
- (C) `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`
- (D) `cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);`

Solution: (C) `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`
`cudaMemcpy` has the following arguments: `cudaMemcpy(destination, source, size, direction)`. In this case, we want to copy 3000 bytes of data from the host array `A_h` to the device array `A_d`. The appropriate API call for this data copy in CUDA would be `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`.

## Exercise 2.8
How would one declare a variable `err` that can appropriately receive the returned value of a CUDA API call?

- (A) `int err;`
- (B) `cudaError err;`
- (C) `cudaError_t err;`
- (D) `cudaSuccess_t err;`

Solution: (C) `cudaError_t err;`

## Exercise 2.9
Consider the following CUDA kernel and the corresponding host function that calls it:
   ```cpp
   /*1*/__global__ void foo_kernel(float* a, float* b, unsigned int N){
   /*2*/    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   /*3*/    if(i < N) {
   /*4*/        b[i] = 2.7f * a[i] - 4.3f;
   /*5*/    }
   /*6*/}
   /*7*/
   /*8*/void foo(float* a_d, float* b_d) {
   /*9*/    unsigned int N = 200000;
   /*10*/    foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
   /*11*/}
   ```
- a. What is the number of threads per block?
- b. What is the number of threads in the grid?
- c. What is the number of blocks in the grid?
- d. What is the number of threads that execute the code on line 02?
- e. What is the number of threads that execute the code on line 04?

Solution:
a. The number of threads per block is 128.
b. (200000 + 128 - 1)/128 = 1563 blocks, and thus the number of threads in the grid is 1563 * 128 = 200064.
c. The number of blocks in the grid is 1563.
d. All threads in the grid execute the code on line 02 i.e. 200064 threads.
e. The number of threads that execute the code on line 04 is 200000 threads.

## Exercise 2.10
A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?

Solution: We can use both `__host__` and `__device__` qualifiers to declare a function that can be executed on both the host and the device. By using these qualifiers, we can declare a single function that can be executed on both the host and the device without having to declare it twice.