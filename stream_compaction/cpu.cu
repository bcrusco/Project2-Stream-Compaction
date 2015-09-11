#include <cstdio>
#include <cstdlib>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	odata[0] = 0;
	for (int i = 1; i < n; i++) {
		odata[i] = odata[i - 1] + idata[i - 1];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int j = 0;
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			odata[j] = idata[i];
			j++;
		}
	}

    return j;
}

void zeroArray(int n, int *a) {
	for (int i = 0; i < n; i++) {
		a[i] = 0;
	}
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int *temp = (int*)malloc(n * sizeof(int));
	zeroArray(n, temp);
	int *scan_output = (int*)malloc(n * sizeof(int));
	zeroArray(n, scan_output);

	// Compute temporary array
	for (int i = 0; i < n; i++) {
		if (idata[i] != 0) {
			temp[i] = 1;
		}
	}

	// Run exclusive scan on the temporary array
	scan(n, scan_output, temp);

	// Scatter
	for (int i = 0; i < n; i++) {
		if (temp[i] == 1) {
			odata[scan_output[i]] = idata[i];
		}
	}

    return scan_output[n - 1] + 1;
}

}
}
