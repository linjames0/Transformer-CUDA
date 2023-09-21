#include <stdio.h>
#include <stdlib.h>

__global__ void matMul(float *d_A, float *d_B, float *d_C, int M, int N, int P) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if(row < M && col < P) {
                float sum = 0.0f;

                // compute the dot product for each row of A and col of B
                for(int i = 0; i < N; ++i) {
                        sum += d_A[row * N + i] * d_B[i * P + col];
                }
                d_C[row * P + col] = sum;
        }
}

int main() {
        // variable initialization
        int M = 2;
        int N = 3;
        int P = 5;
      
        float *h_A, *h_B, *h_C;
        float *d_A, *d_B, *d_C;
      
        // memory allocation
        h_A = (float *)malloc(M * N * sizeof(float));
        h_B = (float *)malloc(N * P * sizeof(float));
        h_C = (float *)malloc(M * P * sizeof(float));
      
        cudaMalloc((void**)&d_A, M * N * sizeof(float));
        cudaMalloc((void**)&d_B, N * P * sizeof(float));
        cudaMalloc((void**)&d_C, M * P * sizeof(float));
      
        // initial data
        for(int i = 0; i < M; ++i) {
                for(int j = 0; j < N; ++j) {
                        h_A[i * N + j] = (float) (rand() % 10 + 1);
                }
        }
      
        for(int i = 0; i < N; ++i) {
                for(int j = 0; j < P; ++j) {
                        h_B[i * P + j] = (float) (rand() % 10 + 1);
                }
        }
      
        // copy CPU data to GPU memory blocks
        cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, N * P * sizeof(float), cudaMemcpyHostToDevice);
      
        // set grid and block dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim((P + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y);
      
        // run matmul
        matMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, P);
      
        // transfer data from device to host
        cudaMemcpy(h_C, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);
      
        // print statements
        printf("Matrix A:\n--------\n");
        for(int i = 0; i < M; ++i) {
                for(int j = 0; j < N; ++j) {
                        printf("%f ", h_A[i * N + j]);
                }
                printf("\n");
        }
      
        printf("--------\n");
        printf("Matrix B:\n--------\n");
        for(int i = 0; i < N; ++i) {
                for(int j = 0; j < P; ++j) {
                        printf("%f ", h_B[i * P + j]);
                }
                printf("\n");
        }
        
        printf("--------\n");
        printf("Matrix C:\n--------\n");
        for(int i = 0; i < M; ++i) {
                for(int j = 0; j < P; ++j) {
                        printf("%f ", h_C[i * P + j]);
                }
                printf("\n");
        }
        printf("--------\n");
      
        // clean up device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
      
        free(h_A);
        free(h_B);
        free(h_C);
      
        return 0;
}
