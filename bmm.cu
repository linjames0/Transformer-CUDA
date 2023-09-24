#include <stdio.h>
#include <stdlib.h>

__global__ void bmm(float *d_A, float *d_B, float *d_C, int batch_size, int M, int N, int P) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.z * blockDim.z + threadIdx.z;

	if (batch < batch_size && row < M && col < P) {
		float sum = 0.0f;
		
                // compute the dot product for each row of A and col of B
                for (int i = 0; i < N; ++i) {
                        sum += d_A[batch * M * N + row * N + i] * d_B[batch * N * P + i * P + col];
                }
		d_C[batch * M * P + row * P + col] = sum;
	}
}

int main() {
	// variable initialization
        int M = 2;
        int N = 3;
        int P = 5;
	int batch_size = 4;	

        float *h_A, *h_B, *h_C;
        float *d_A, *d_B, *d_C;
      
        // memory allocation
        h_A = (float *)malloc(batch_size * M * N * sizeof(float));
        h_B = (float *)malloc(batch_size * N * P * sizeof(float));
        h_C = (float *)malloc(batch_size * M * P * sizeof(float));
      
        cudaMalloc((void**)&d_A, batch_size * M * N * sizeof(float));
        cudaMalloc((void**)&d_B, batch_size * N * P * sizeof(float));
        cudaMalloc((void**)&d_C, batch_size * M * P * sizeof(float));
      
        // initial data
	for (int batch = 0; batch < batch_size; ++batch) {
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				h_A[batch * M * N + i * N + j] = (float) (rand() % 10 + 1);
			}
		}
	      
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < P; ++j) {
				h_B[batch * N * P + i * P + j] = (float) (rand() % 10 + 1);
			}
		}
	}
      
        // copy CPU data to GPU memory blocks
        cudaMemcpy(d_A, h_A, batch_size * M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, batch_size * N * P * sizeof(float), cudaMemcpyHostToDevice);
      
        // set grid and block dimensions
        dim3 blockDim(8, 8, batch_size);
        dim3 gridDim((P + blockDim.x - 1)/blockDim.x, (M + blockDim.y - 1)/blockDim.y, (batch_size + blockDim.z - 1)/blockDim.z);
      	
        // run batch matmul
        bmm<<<gridDim, blockDim>>>(d_A, d_B, d_C, batch_size, M, N, P);
      
        // sync
	cudaDeviceSynchronize();
	
	// check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    		printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}

	// transfer data from device to host
        cudaMemcpy(h_C, d_C, batch_size * M * P * sizeof(float), cudaMemcpyDeviceToHost);
      
        // print statements
	for (int batch = 0; batch < batch_size; ++batch) {
		printf("Matrix A:\n--------\n");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				printf("%f ", h_A[batch * M * N + i * N + j]);
			}
			printf("\n");
		}
	      
		printf("--------\n");
		printf("Matrix B:\n--------\n");
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < P; ++j) {
				printf("%f ", h_B[batch * N * P + i * P + j]);
			}
			printf("\n");
		}
		
		printf("--------\n");
		printf("Matrix C:\n--------\n");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < P; ++j) {
				printf("%f ", h_C[batch * M * P + i * P + j]);
			}
			printf("\n");
		}
		printf("--------\n");
	}
	
        // clean up device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
      
        free(h_A);
        free(h_B);
        free(h_C);
      
        return 0;
}
