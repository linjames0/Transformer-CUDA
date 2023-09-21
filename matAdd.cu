#include <stdio.h>
#include <stdlib.h>

// matrix addition kernel
__global__ void matAdd(float *d_A, float *d_B, float *d_C, int M, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// add matrix elements
	if (row < M && col < N) {
		d_C[row * N + col] = d_A[row * N + col] + d_B[row * N + col];
	}
}

int main() {
  	// var declaration
	int N = 5;
	int M = 5;
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;

	// allocate host memory
	h_A = (float *)malloc(M * N * sizeof(float));
	h_B = (float *)malloc(M * N * sizeof(float));
        h_C = (float *)malloc(M * N * sizeof(float));

	// allocate device memory
	cudaMalloc(&d_A, M * N * sizeof(float));
	cudaMalloc(&d_B, M * N * sizeof(float));
	cudaMalloc(&d_C, M * N * sizeof(float));

	// initialize data
	for (int i = 0; i < M * N; ++i) {
		h_A[i] = i - 3;
		h_B[i] = i;
	}
	
	// copy host data to device
	cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

	// launch kernel instance
	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y);
	matAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);
	
	// copy result back to host
	cudaMemcpy(h_A, d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  	// display results
	printf("Matrix A: \n");
	printf("----------\n");
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("A: %f ", h_A[i * N + j]);
		}
		printf("\n)
	}

	printf("----------\n");
	printf("Matrix B: \n");
	printf("----------\n");
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("B: %f ", h_B[i * N + j]);
		}
		printf("\n)
	}

	printf("----------\n");
	printf("Matrix C: \n");
	printf("----------\n");
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("C: %f ", h_C[i * N + j]);
		}
		printf("\n)
	}

	// clean up data
	free(h_A); free(h_B); free(h_C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

	return 0;
}
