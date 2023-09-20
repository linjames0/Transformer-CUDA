#include <stdio.h>
#include <stdlib.h>

// scale kernel
__global__ void matScale(float *d_A, float *d_B, float scale, int N, int M) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// scale matrix elements
	if (row < N && col < M) {
		d_B[row * M + col] = d_A[row * M + col] / scale;
	}
}

int main() {
  	// var declaration
	int N = 3;
	int M = 3;
	float scale = 2.0f;
	float *A, *B;
	float *d_A, *d_B;

	// allocate host memory
	A = (float *)malloc(N * M * sizeof(float));
	B = (float *)malloc(N * M * sizeof(float));

	// allocate device memory
	cudaMalloc(&d_A, N * M * sizeof(float));
	cudaMalloc(&d_B, N * M * sizeof(float));

	// initialize data
	for (int i = 0; i < N * M; ++i) {
		A[i] = i - 3;
		B[i] = i;
	}
	
	// copy host data to device
	cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);
        
	// launch kernel instance
	dim3 blockDim(16, 16);
	dim3 gridDim((M + blockDim.x - 1)/blockDim.x, (N + blockDim.y - 1)/blockDim.y);
	matScale<<<gridDim, blockDim>>>(d_A, d_B, scale, N, M);
	
	// copy result back to host
	cudaMemcpy(A, d_A, N * M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost);

  	// display results
	printf("Matrix A: \n");
	printf("----------\n");
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j) {
			printf("A: %f ", A[i * M + j]);
		}
		printf("\n");
	}

	printf("----------\n");
	printf("Matrix B: \n");
	printf("----------\n");
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j) {
			printf("B: %f ", B[i * M + j]);
		}
		printf("\n");
	}
	
	// clean up data
	free(A); free(B);
	cudaFree(d_A); cudaFree(d_B);

	return 0;
}
