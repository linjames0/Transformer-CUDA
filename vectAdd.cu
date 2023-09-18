#include <stdio.h>
#include <stdlib.h>

__global__ void addVectors(float *d_a, float *d_b, float *d_c, int N) {
        int i = threadIdx.x;

        if(i < N) {
                d_c[i] = d_a[i] + d_b[i];
        }
}

int main() {
        int N = 100;
        float *a, *b, *c;
        float *d_a, *d_b, *d_c;

        // allocate host memory
        a = (float *)malloc(N * sizeof(float));
        b = (float *)malloc(N * sizeof(float));
        c = (float *)malloc(N * sizeof(float));

        // allocate device memory
        cudaMalloc(&d_a, N * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_c, N * sizeof(float));

        // initialize data
        for(int i = 0; i < N; ++i) {
                a[i] = i - 3;
                b[i] = i;
        }

        // copy host data to device
        cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, c, N * sizeof(float), cudaMemcpyHostToDevice);

        // kernel launch: vector addition
        addVectors<<<1, N>>>(d_a, d_b, d_c, N);

        // copy result back to host
        cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

        for(int i = 0; i < 6; ++i) {
                printf("a: %f b: %f c: %f ", a[i], b[i], c[i]);
                printf("\n");
        }

        // clean up data
        free(a); free(b); free(c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

        return 0;
}
