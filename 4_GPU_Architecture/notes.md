# GPU Architecture

- A GPU consists of multiple **Streaming Multirocessors (SMs)**.
- Each SM consists of multiple **CUDA cores**. A core is the unit that executes arithmetic operations.
- These cores in SM shares a memory called **shared memory**.
- These cores in SM also share a **control logic**.
- All the SMs have access to a **global memory**.
- Global memory is where we have been copying data from host (CPU) to device (GPU) using `cudaMemcpy`.
 
 ![SM architecture](images/gpu_architecture.png)

- An NVIDIA Tesla T4 GPU has 40 SMs and each SM has 64 CUDA cores. So in total we have 2560 CUDA cores.

## Assigning work to GPU
- Threads are grouped in block (1D, 2D, 3D - Discussed in previous chapter).
- Recall that a block is a logical group of threads and we use `<<<num_blocks, num_threads_per_block>>>` syntax to launch a kernel.
- Threads in a block are executed by a single SM. In other words, all threads in a block are executed by a single SM.
- Note that we can have multiple thread blocks get assigned to same SM. However we can't have some threads in one block assigned to one SM and some threads in another block assigned to another SM.
- Threads when run on SM, they require resources e.g. registers, memory, some slots for thread specific control data etc
- So SMs can only run a limited number of threads/blocks at a time and thus we only have a limited number of thread blocks getting executed simultaneously.
- When we launch a kernel, a grid of threads is launched. A grid contain thread blocks. The number of blocks in the grid can exceed the total number of blocks that can be executed simultaneously. In that case, the remaining blocks wait for other blocks to finish execution before they can be assigned to an SM.

## Syncronization
- Threads in the same block can collaborate and communicate in ways that threads in different blocks cannot.
- Threads in the same block can synchronize using `__syncthreads()` function. This is called **barrier synchronization**.
    -  Wait for all threads in a block to reach the barrier before any of them proceeds past it.
- Another way the threads in same block collaborate is by using **shared memory**. (To be elaborated later)
    - Access to a fast memory shared by all threads in the same block.

## Scheduling Considerations
- As discussed earlier, threads in a block are executed by a single SM. This makes supporting collaboration between threads efficient.
- Also, all threads in a block assigned to SM simultaneously. 
- A block cannot be assigned to an SM if all the resources required by the block are not available.
    - So if an SM has 256 slots free, but there are 512 threads in a block, that means we can't run that block on that SM.
    - If it is allowed, we can have deadlock. For example if 256 of the 512 threads in the block are executing, while other 256 are waiting to be scheduled on the same SM. If first 256 threads reach a barrier, they will wait for other 256 threads to reach the barrier. On the other hand, other 256 threads are waiting to be scheduled on the SM and may be waiting for the first 256 threads to finish. This is a deadlock.

## Transparent Scalability
- Threads in different blocks do not synchronize. In other words, threads in different blocks do not collaborate. Blocks are independent of each other.
- This allows blocks to execute in any order and on any SM.
- This also means blocks can execute in parallel or sequentially with respect to each other.
- This allows different GPU architectures to execute the same code with different amounts of parallelism.
    - If device has more SMs, more blocks can be executed simultaneously.
    - If device has less SMs, blocks will be executed sequentially.
    - This is called **transparent scalability**.
- Programmers don't have to worry about how many SMs are there in the device. They just write the code and the code will run on any device with any number of SMs.
- Programmers often don't need to worry about the architecture of CPUs too but for different reasons. In CPUs threads can synchronize and collaborate. CPU allows context switching between threads. However due to massive parallelism in GPUs, context switching is not feasible. Hence, the restriction on threads in different blocks not to synchronize.

## Thread Scheduling
Now, lets focus on what happens on one SM when a block is assigned to it.
- Each SM has a **scheduler** that manages the execution of threads.
- Threads assigned to an SM run concurrently (not that it doesn't mean they run in parallel, it means that scheduler switches between threads to give an illusion of parallelism).
- The scheduler selects a block to execute and then selects a **warp** (a group of 32 threads) from that block.

### Warps
- Warps are the unit of scheduling and execution on the SM.
- Size of the warps is device specific, although it has always been 32 threads till now.
- Threds in a warp are scheduled together and executed following **SIMD (Single Instruction, Multiple Data)** model.
- This means that all threads in a warp execute the same instruction at the same time however all these threads will be working on different data.
- The advantage of that is instruction fetch and decode is done only once for the warp. This reduces amount of control logic required and thus saves power and real estate.

### Processing Blocks
- SM can also be divided into multiple **processing blocks**. Each processing block has its own scheduler, register file, instruction cache, dispatch unit etc.
- For example, an SM with 64 CUDA cores can be divided into 4 processing blocks each with 16 CUDA cores (variable across different architectures).
- A warp is executed by a single processing block. In other words, all threads in a warp are executed by a single processing block.
- This doesn't mean processing block executes only one warp at a time. It can execute multiple warps but each warp is executed by a single processing block.




