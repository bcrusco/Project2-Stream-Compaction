#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	thrust::host_vector<int> hst_in(idata, idata + n);
	thrust::device_vector<int> dev_in = hst_in;
	thrust::device_vector<int> dev_out(n);

	thrust::exclusive_scan(dev_in.begin(), dev_in.end(), dev_out.begin());
	thrust::host_vector<int> hst_out = dev_out;

	for (int i = 0; i < n; i++) {
		odata[i] = hst_out[i];
	}
}

}
}
