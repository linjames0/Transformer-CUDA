# Overview

This repository contains a collection of CUDA programs that perform various mathematical operations on matrices and vectors. These operations include matrix multiplication, matrix scaling, softmax function implementation, vector addition, matrix addition, and dot product calculation. The programs are designed to leverage the parallel processing capabilities of GPUs to perform these operations more efficiently than traditional CPU-based implementations.

The programs are written in C and use CUDA for GPU programming. They define kernel functions that perform the operations on the GPU, and main functions that handle memory allocation, data initialization, data transfer between the host and device, kernel launching, result printing, and memory cleanup. The programs also include necessary header files and use random data for testing.

# Technologies and Frameworks

- CUDA
- C programming language
- Nvidia CUDA kernel
- Parallel processing
- GPU programming
- Matrix and vector operations
- Softmax function
- Dot product calculation
- Parallel reduction
# Installation

This guide will walk you through the steps required to install and run the project.

## Prerequisites

Before you start, ensure that you have the following:

- A C compiler that supports the standard library headers `<stdio.h>` and `<stdlib.h>`.
- The CUDA toolkit, which includes the CUDA runtime library and the necessary headers for GPU programming.
- A compatible GPU device with CUDA support.
- Sufficient memory allocation for the host and device arrays used in the program.

## Steps

1. **Install the C compiler**

   You can download the C compiler from the official website. Follow the instructions provided to install it on your system.

2. **Install the CUDA toolkit**

   The CUDA toolkit can be downloaded from the official NVIDIA website. Follow the instructions provided to install it on your system.

3. **Check your GPU compatibility**

   Ensure that your GPU device supports CUDA. You can check this on the official NVIDIA website.

4. **Clone the repository**

   Clone the repository to your local machine using the following command:

   ```
   git clone https://github.com/username/repository.git
   ```

5. **Compile the CUDA programs**

   Navigate to the directory containing the CUDA programs and compile them using the `nvcc` compiler. For example:

   ```
   nvcc matMul.cu -o matMul
   ```

6. **Run the CUDA programs**

   After compiling the CUDA programs, you can run them using the following command:

   ```
   ./matMul
   ```

Please note that you need to replace `matMul` with the name of the program you want to compile and run.

## Troubleshooting

If you encounter any issues during the installation process, ensure that you have correctly installed the C compiler and the CUDA toolkit, and that your GPU device is compatible with CUDA. Also, check that you have sufficient memory allocation for the host and device arrays used in the program.

# Usage

This section provides a few basic examples of how to use the functionality of the project. 

## vectAdd.cu

The `vectAdd.cu` file is used for vector addition. Here is a basic example of how to use it:

```c
// Initialize vectors A and B
float* A = (float*)malloc(size);
float* B = (float*)malloc(size);

// Fill vectors with some values
for(int i = 0; i < N; i++){
    A[i] = i;
    B[i] = i;
}

// Call the function from vectAdd.cu
vectAdd(A, B, N);
```

## matAdd.cu

The `matAdd.cu` file is used for matrix addition. Here is a basic example of how to use it:

```c
// Initialize matrices A and B
float* A = (float*)malloc(size);
float* B = (float*)malloc(size);

// Fill matrices with some values
for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
        A[i*N + j] = i + j;
        B[i*N + j] = i - j;
    }
}

// Call the function from matAdd.cu
matAdd(A, B, N);
```

## dotProd.cu

The `dotProd.cu` file is used for calculating the dot product of two vectors. Here is a basic example of how to use it:

```c
// Initialize vectors A and B
float* A = (float*)malloc(size);
float* B = (float*)malloc(size);

// Fill vectors with some values
for(int i = 0; i < N; i++){
    A[i] = i;
    B[i] = i;
}

// Call the function from dotProd.cu
float result = dotProd(A, B, N);
```

Please note that the above examples are simplified for the sake of clarity. In a real-world scenario, you would need to handle memory allocation and deallocation, error checking, and possibly other aspects depending on your specific use case.
