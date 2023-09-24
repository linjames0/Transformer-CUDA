#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void mean(float *d_a, float *d_mean, float *redArr1, int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < N) {
		redArr1[col] = d_a[col];
	}

	__syncthreads();

	// determine amount of padding for parallel reduction
	int padding = 0;
	for (int e = 0; (float)N/(float)(1 << e)>= 1; ++e) {
	      	padding = e + 1;
	}

	for (int i = 0; i < (1 << padding) - N; ++i) {
		redArr1[N + i] = 0;
	}

	__syncthreads();

	// sum using parallel reduction
	for (int stride = 1 << padding; stride >= 1; stride /= 2) {
		if (col < stride) {
			redArr1[col] += redArr1[col + stride];
		}
	}
	
	__syncthreads();

	d_mean[0] = redArr1[0] / N;
}

__global__ void var(float *d_a, float *d_var, float *d_mean, float *redArr1, float *redArr2, int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	mean<<<1, N>>>(d_a, d_mean, redArr1, N);
	
	cudaDeviceSynchronize();

	if (col < N) {
		redArr2[col] = powf(d_a[col] - d_mean[0], 2); 
	}

	// determine amount of padding for parallel reduction
        int padding = 0;
        for (int e = 0; (float)N/(float)(1 << e)>= 1; ++e) {
                padding = e + 1;
        }

        for (int i = 0; i < (1 << padding) - N; ++i) {
                redArr2[N + i] = 0;
        }

        __syncthreads();

        // sum using parallel reduction
        for (int stride = 1 << padding; stride >= 1; stride /= 2) {
                if (col < stride) {
                        redArr2[col] += redArr2[col + stride];
                }
        }

        __syncthreads();

        d_var[0] = redArr2[0] / (N - 1);
}

__global__ void layernorm(float *d_a, float *d_norm, float *d_var, float *d_mean, float *redArr1, float *redArr2, int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	mean<<<1, N>>>(d_a, d_mean, redArr1, N);
	cudaDeviceSynchronize();
	var<<<1, N>>>(d_a, d_var, d_mean, redArr1, redArr2, N);
	cudaDeviceSynchronize();

	if (col < N) {
		d_norm[col] = (d_a[col] - d_mean[0]) / sqrtf(d_var[0] + 1e-8);
	}
}



int main() {
	// var declaration
	int N = 5;
	float *h_a, *h_mean, *h_var, *h_norm;
	float *d_a, *d_mean, *d_var, *d_norm;
	float *redArr1, *redArr2;

	// memory allocation
	h_a = (float *)malloc(N * sizeof(float));
	h_mean = (float *)malloc(1 * sizeof(float));
        h_var = (float *)malloc(1 * sizeof(float));
	h_norm = (float *)malloc(N * sizeof(float));

	cudaMalloc((void**)&d_a, N * sizeof(float));
	cudaMalloc((void**)&d_mean, 1 * sizeof(float));
        cudaMalloc((void**)&d_var, 1 * sizeof(float));
	cudaMalloc((void**)&d_norm, N * sizeof(float));
	cudaMalloc((void**)&redArr1, 2 * N * sizeof(float));
        cudaMalloc((void**)&redArr2, 2 * N * sizeof(float));

	// populate vectors with data	
	for (int i = 0; i < N; ++i) {
		h_a[i] = (float) (rand() % 10 + 1);
	}

	cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
	
	// launch kernel instance
	layernorm<<<1, N>>>(d_a, d_norm, d_var, d_mean, redArr1, redArr2, N);
	
	// copy results to CPU
	cudaMemcpy(h_mean, d_mean, 1 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_var, d_var, 1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_norm, d_norm, N * sizeof(float), cudaMemcpyDeviceToHost);

	// print results
	printf("A:\n--------\n");
        for(int i = 0; i < N; ++i) {
                printf("%f ", h_a[i]);
                printf("\n");
        }
      
        printf("--------\n");
        printf("Mean:\n");
        printf("%f ", h_mean[0]);
        printf("\n--------\n");

	printf("--------\n");
        printf("Var:\n");
        printf("%f ", h_var[0]);
        printf("\n--------\n");

	printf("Norm:\n--------\n");
        for(int i = 0; i < N; ++i) {
                printf("%f ", h_norm[i]);
                printf("\n");
        }

	// clean up memory
	cudaFree(d_a);
	cudaFree(d_mean);
	cudaFree(d_var);
	cudaFree(d_norm);
	cudaFree(redArr1);
	cudaFree(redArr2);

	free(h_a);
	free(h_mean);
	free(h_var);
	free(h_norm);

	return 0;
}
