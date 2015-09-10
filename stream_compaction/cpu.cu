#include <cstdio>
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

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO
    return -1;
}

}
}
