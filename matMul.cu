#include <stdio.h>
#include <stdlib.h>

__global__ void matMul(float *d_A, float *d_B, float *d_C, int N, int M, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if(row < N && col < K) {
                float sum = 0.0f;

                // compute the dot product for each row of A and col of B
                for(int i = 0; i < M; ++i) {
                        sum += d_A[row * M + i] * d_B[i * K + col];
                }
                d_C[row * K + col] = sum;
        }
}

int main() {
        // variable initialization
        int N = 2;
        int M = 3;
        int K = 5;
      
        float *h_A, *h_B, *h_C;
        float *d_A, *d_B, *d_C;
      
        // memory allocation
        h_A = (float *)malloc(N * M * sizeof(float));
        h_B = (float *)malloc(M * K * sizeof(float));
        h_C = (float *)malloc(N * K * sizeof(float));
      
        cudaMalloc((void**)&d_A, N * M * sizeof(float));
        cudaMalloc((void**)&d_B, M * K * sizeof(float));
        cudaMalloc((void**)&d_C, N * K * sizeof(float));
      
        // initial data
        for(int i = 0; i < N; ++i) {
                for(int j = 0; j < M; ++j) {
                        h_A[i * M + j] = (float) (rand() % 10 + 1);
                }
        }
      
        for(int i = 0; i < M; ++i) {
                for(int j = 0; j < K; ++j) {
                        h_B[i * K + j] = (float) (rand() % 10 + 1);
                }
        }
      
        // copy CPU data to GPU memory blocks
        cudaMemcpy(d_A, h_A, N * M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, M * K * sizeof(float), cudaMemcpyHostToDevice);
      
        // set grid and block dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim((K + blockDim.x - 1)/blockDim.x, (N + blockDim.y - 1)/blockDim.y);
      
        // run matmul
        matMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, M, K);
      
        // transfer data from device to host
        cudaMemcpy(h_C, d_C, N * K * sizeof(float), cudaMemcpyDeviceToHost);
      
        // print statements
        printf("Matrix A:\n--------\n");
        for(int i = 0; i < N; ++i) {
                for(int j = 0; j < M; ++j) {
                        printf("%f ", h_A[i * M + j]);
                }
                printf("\n");
        }
      
        printf("--------\n");
        printf("Matrix B:\n--------\n");
        for(int i = 0; i < M; ++i) {
                for(int j = 0; j < K; ++j) {
                        printf("%f ", h_B[i * K + j]);
                }
                printf("\n");
        }
        
        printf("--------\n");
        printf("Matrix C:\n--------\n");
        for(int i = 0; i < N; ++i) {
                for(int j = 0; j < K; ++j) {
                        printf("%f ", h_C[i * K + j]);
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
