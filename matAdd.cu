#include <stdio.h>
#include <stdlib.h>

// matrix addition kernel
__global__ void matAdd(float *d_A, float *d_B, float *d_C, int N, int M) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(row < N && col < M) {
		d_C[row * M + col] = d_A[row * M + col] + d_B[row * M + col];
	}
}

int main() {
  // var declaration
	int N = 5;
	int M = 5;
	float *A, *B, *C;
	float *d_A, *d_B, *d_C;

	// allocate host memory
	A = (float *)malloc(N * M * sizeof(float));
	B = (float *)malloc(N * M * sizeof(float));
        C = (float *)malloc(N * M * sizeof(float));

	// allocate device memory
	cudaMalloc(&d_A, N * M * sizeof(float));
	cudaMalloc(&d_B, N * M * sizeof(float));
	cudaMalloc(&d_C, N * M * sizeof(float));

	// initialize data
	for(int i = 0; i < N * M; ++i) {
		A[i] = i - 3;
		B[i] = i;
	}
	
	// copy host data to device
	cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, N * M * sizeof(float), cudaMemcpyHostToDevice);
	
	// kernel launch: vector addition
	dim3 blockDim(16, 16);
	dim3 gridDim((M + blockDim.x - 1)/blockDim.x, (N + blockDim.y - 1)/blockDim.y);
	addVectors<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, M);
	
	// copy result back to host
	cudaMemcpy(A, d_A, N * M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

  // display results
	for(int i = 0; i < N * M; ++i) {
		printf("A: %f B: %f C: %f ", A[i], B[i], C[i]);
		printf("\n");
	}

	// clean up data
	free(A); free(B); free(C);
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

	return 0;
}