#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

__global__ void up_sweep(int d, int *data) {
	int k = threadIdx.x;
	int p2d = pow(2.0, (double)d);
	int p2da1 = pow(2.0, (double)(d + 1));

	if (k % p2da1 == 0) {
		data[k + p2da1 - 1] += data[k + p2d - 1];
	}
}

__global__ void down_sweep(int d, int *data) {
	int k = threadIdx.x;
	int p2d = pow(2.0, (double)d);
	int p2da1 = pow(2.0, (double)(d + 1));

	if (k % p2da1 == 0) {
		int temp = data[k + p2d - 1];
		data[k + p2d - 1] = data[k + p2da1 - 1];
		data[k + p2da1 - 1] += temp;
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

	// Can use one array for input and output in this implementation
	int *dev_data;
	cudaMalloc((void**)&dev_data, m * sizeof(int));
	cudaMemcpy(dev_data, new_idata, m * sizeof(int), cudaMemcpyHostToDevice);

	// Execute scan on device
	for (int d = 0; d < ilog2ceil(n); d++) {
		up_sweep<<<1, m>>>(d, dev_data);
	}

	cudaMemset((void*)&dev_data[m - 1], 0, sizeof(int));
	for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
		down_sweep<<<1, m>>>(d, dev_data);
	}

	cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_data);
	free(new_idata);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    // TODO
    return -1;
}

}
}
