#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

__global__ void kern_scan(int d, int *idata, int *odata) {
	int k = threadIdx.x;

	if (k >= (int)pow(2.0, (double)(d - 1))) {
		odata[k] = idata[k - (int)pow(2.0, (double)(d - 1))] + idata[k];
	}
	else {
		odata[k] = idata[k];
	}
}

void padArrayRange(int start, int end, int *a) {
	for (int i = start; i < end; i++) {
		a[i] = 0;
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	int m = pow(2, ilog2ceil(n));
	int *new_idata = (int*)malloc(m * sizeof(int));

	// Expand array to next power of 2 size
	for (int i = 0; i < n; i++) {
		new_idata[i] = idata[i];
	}
	padArrayRange(n, m, new_idata);

	int *dev_idata;
	int *dev_odata;

	cudaMalloc((void**)&dev_idata, m * sizeof(int));
	cudaMemcpy(dev_idata, new_idata, m * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_odata, m * sizeof(int));

	// Execute scan on device
	for (int d = 1; d <= ilog2ceil(n); d++) {
		kern_scan<<<1, m>>>(d, dev_idata, dev_odata);
		dev_idata = dev_odata;
	}

	odata[0] = 0;
	cudaMemcpy(odata + 1, dev_odata, (m * sizeof(int)) - sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_idata);
	cudaFree(dev_odata);
	free(new_idata);
}

}
}


