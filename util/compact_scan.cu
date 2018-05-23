#include "scan.h"
#include "gutil.h"

namespace gpsm {
namespace scan {
	//---------------------------------------------------------------------------
	__global__ void scatterInner(	bool *in, 
									int *out, 
									int len) 
	{
		uint index = GTID;
		if (index < len) {
			out[index] = in[index] == false ? 0 : 1;
		}
	}
	//---------------------------------------------------------------------------
	__global__ void compactInner(	bool *in, 
									int *out, 
									int *indices, 
									int len) 
	{
		uint index = GTID;
		if (index < len) {
			if (in[index] != 0) {
				int pos = indices[index];
				out[pos] = index;
			}
		}
	}
	//---------------------------------------------------------------------------
	int scatter(bool *dev_in, 
				int *dev_offset, 
				int numRecords)
	{
		uint blocksPerGrid = (numRecords + BLOCK_SIZE - 1) / BLOCK_SIZE;
		const dim3 TPB(BLOCK_SIZE);

		int *dev_tmp;
		CUDA_SAFE_CALL(cudaMalloc(&dev_tmp, numRecords * sizeof(int)));

		// convert a boolean array to an int array
		scatterInner <<< blocksPerGrid, BLOCK_SIZE >>>(dev_in, dev_tmp, numRecords);
		CUDA_SAFE_CALL(cudaGetLastError());

		// find output offsets by using prefix sum
		int sum = prefixSum(dev_tmp, dev_offset, numRecords);

		CUDA_SAFE_CALL(cudaFree(dev_tmp));
		return sum;
	}
	//---------------------------------------------------------------------------
	int compact(bool* d_inArr, 
				int* d_outArr, 
				int numRecords) 
	{
		uint blocksPerGrid = (numRecords + BLOCK_SIZE - 1) / BLOCK_SIZE;
		
		int* d_offsetArr;
		CUDA_SAFE_CALL(cudaMalloc(&d_offsetArr, numRecords * sizeof(int)));
		
		int sum = scatter(d_inArr, d_offsetArr, numRecords);
		
		if (sum != 0) {
			compactInner <<< blocksPerGrid, BLOCK_SIZE >>>(d_inArr, d_outArr, d_offsetArr, numRecords);
			CUDA_SAFE_CALL(cudaGetLastError());
		}

		CUDA_SAFE_CALL(cudaFree(d_offsetArr));
		return sum;
	}
}}