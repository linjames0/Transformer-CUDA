#include <stdio.h>
#include <stdlib.h>

// scale kernel
__global__ void matScale(float *d_A, float *d_B, float scale, int M, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// scale matrix elements
	if (row < M && col < N) {
		d_B[row * N + col] = d_A[row * N + col] / scale;
	}
}

int main() {
  	// var declaration
	int M = 3;
	int N = 3;
	float scale = 2.0f;
	float *h_A, *h_B;
	float *d_A, *d_B;

	// allocate host memory
	h_A = (float *)malloc(M * N * sizeof(float));
	h_B = (float *)malloc(M * N * sizeof(float));

	// allocate device memory
	cudaMalloc(&d_A, M * N * sizeof(float));
	cudaMalloc(&d_B, M * N * sizeof(float));

	// initialize data
	for (int i = 0; i < M * N; ++i) {
		h_A[i] = i - 3;
		h_B[i] = i;
	}
	
	// copy host data to device
	cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, M * N * sizeof(float), cudaMemcpyHostToDevice);
        
	// launch kernel instance
	dim3 blockDim(16, 16);
	dim3 gridDim((N + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y);
	matScale<<<gridDim, blockDim>>>(d_A, d_B, scale, M, N);
	
	// copy result back to host
	cudaMemcpy(h_A, d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, M * N * sizeof(float), cudaMemcpyDeviceToHost);

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
	printf("Matrix B: \n");
	printf("----------\n");
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("B: %f ", h_B[i * N + j]);
		}
		printf("\n");
	}
	
	// clean up data
	free(h_A); free(h_B);
	cudaFree(d_A); cudaFree(d_B);

	return 0;
}
