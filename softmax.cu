#include <stdio.h>
#include <stdlib.h>

__global__ void softmax(float *d_in, float *d_out, float *expArr, float *redArr, int N) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < N) {
		float local_exp = expf(d_in[col]);
		expArr[col] = local_exp;
		redArr[col] = expArr[col];
		
		// amount of padding for parallel reduction
	        int padding = 0;
        	for (int e = 0; (float)N/(float)(1 << e)>= 1; ++e) {
                	padding = e + 1;
        	}

        	for (int i = 0; i < (1 << padding) - N; ++i) {
                	expArr[N + i] = 0;
			redArr[N + i] = 0;
        	}


		__syncthreads();	
		
		// parallel reduction to compute sum
		for (int stride = 1 << padding; stride >= 1; stride /= 2) {
			if (col < stride) {
				redArr[col] += redArr[col + stride];
			}
		}
		__syncthreads();
	}

	// calculate e^x / sum(e^x) = softmax for each element
	if (col < N) {
		float sum = redArr[0];
		d_out[col] = expArr[col] / sum;
	}
}

int main() {
	// var declaration
	int N = 6;
	float h_in[N];
	float h_out[N];
	float *d_in, *d_out;
	float *expArr;
	float *redArr;

	// memory allocation
	cudaMalloc((void**)&d_in, N * sizeof(float));
	cudaMalloc((void**)&d_out, N * sizeof(float));
	cudaMalloc((void**)&expArr, 2 * N * sizeof(float));
	cudaMalloc((void**)&redArr, 2 * N * sizeof(float));

	// data initialization
	for (int i = 0; i < N; ++i) {
		h_in[i] = (float)(rand() % 5 + 1);
	}
	
	cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
	
	// launch softmax kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	softmax<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, expArr, redArr, N);

	// copy result to host
	cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

	// print result
    	printf("Softmax input:\n");
    	for (int i = 0; i < N; ++i) {
        	printf("%f ", h_in[i]);
    	}
    	printf("\n----------\n");
  	printf("Softmax output:\n");
    	for (int i = 0; i < N; ++i) {
        	printf("%f ", h_out[i]);
    	}

    	printf("\n----------\n");

   	// compare with CPU implementation
    	float sum = 0.0f;
    	for (int i = 0; i < N; ++i) {
       	 	sum += exp(h_in[i]);
    	}
    	printf("Expected output:\n");
    	for (int i = 0; i < N; ++i) {
        	printf("%f ", exp(h_in[i]) / sum);
   	}
    	printf("\n");

	// clean device memory
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
