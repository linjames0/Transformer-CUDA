#include <stdio.h>
#include <stdlib.h>

// ReLU kernel
__global__ void relu(float *d_a, float *d_b, float alpha, int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// apply leaky ReLU
	if (col < N) {
		d_b[col] = fmaxf(alpha * d_a[col], d_a[col]);
	}
}

int main() {
  	// var declaration
	int N = 5;
	float *h_a, *h_b;
	float *d_a, *d_b;
	float alpha = 0.01;

	// allocate host memory
	h_a = (float *)malloc(N * sizeof(float));
	h_b = (float *)malloc(N * sizeof(float));

	// allocate device memory
	cudaMalloc(&d_a, N * sizeof(float));
	cudaMalloc(&d_b, N * sizeof(float));

	// initialize data
	for (int i = 0; i < N; ++i) {
		h_a[i] = (float)(rand() % 10 - 4);
	}
	
	// copy host data to device
	cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
        
	// launch kernel instance
	dim3 blockDim(256);
	dim3 gridDim((N + blockDim.x - 1)/blockDim.x);
	relu<<<gridDim, blockDim>>>(d_a, d_b, alpha, N);
	
	// copy result back to host
	cudaMemcpy(h_b, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);

  	// display results
	printf("Vector A: \n");
	printf("----------\n");
	for (int i = 0; i < N; ++i) {
		printf("%f \n", h_a[i]);
	}

	printf("\n----------\n");
	printf("ReLU: \n");
	printf("----------\n");
	for (int i = 0; i < N; ++i) {
		printf("%f \n", h_b[i]);
	}
	
	// clean up data
	free(h_a); free(h_b);
	cudaFree(d_a); cudaFree(d_b);

	return 0;
}
