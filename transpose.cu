#include <stdio.h>
#include <stdlib.h>

__global__ void transpose(float *d_A, float *d_T, int M, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// swap elements via transpose
	if (row < M && col < N) {
		d_T[col * M + row] = d_A[row * N + col];
	}
}

int main() {
  	// var declaration
	int M = 3;
	int N = 4;
	float *h_A, *h_T;
	float *d_A, *d_T;

	// allocate host memory
	h_A = (float *)malloc(M * N * sizeof(float));
	h_T = (float *)malloc(M * N * sizeof(float));

	// allocate device memory
	cudaMalloc(&d_A, M * N * sizeof(float));
	cudaMalloc(&d_T, M * N * sizeof(float));

	// initialize data
	for (int i = 0; i < M * N; ++i) {
		h_A[i] = (float)(rand() % 10 + 1);
	}

	// copy host data to device
	cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, h_T, M * N * sizeof(float), cudaMemcpyHostToDevice);

	// launch kernel instance
	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y);
	transpose<<<gridDim, blockDim>>>(d_A, d_T, M, N);

	// copy result back to host
	cudaMemcpy(h_A, d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_T, d_T, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  	// display results
	printf("Matrix A: \n");
	printf("----------\n");
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("A: %f ", h_A[i * N + j]);
		}
		printf("\n");
	}

	printf("----------\n");
	printf("Transpose: \n");
	printf("----------\n");
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < M; ++j) {
			printf("%f ", h_T[i * M + j]);
		}
		printf("\n");
	}

	// clean up data
	free(h_A); free(h_T);
	cudaFree(d_A); cudaFree(d_T);

	return 0;
}
