#ifndef __GPSM_SCAN_H__
#define __GPSM_SCAN_H__

namespace gpsm {
namespace scan {

	/* Parallel prefix sum on the device */
	int prefixSum(int* d_inArr, int* d_outArr, int numRecords);

	/* Parallel stream compaction on the device */
	int compact(bool* d_inArr, int* d_outArr, int numRecords);
}}

#endif
