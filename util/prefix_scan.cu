#include "scan.h"
#include "gutil.h"
#include <assert.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

namespace gpsm {
namespace scan {
	//---------------------------------------------------------------------------
	template <bool isNP2>
	__device__ void loadSharedChunkFromMem(int *s_data,
										   const int *g_idata, 
										   int n, int baseIndex,
										   int& ai, int& bi, 
										   int& mem_ai, int& mem_bi, 
										   int& bankOffsetA, int& bankOffsetB)
	{
		int thid = threadIdx.x;
		mem_ai = baseIndex + threadIdx.x;
		mem_bi = mem_ai + blockDim.x;

		ai = thid;
		bi = thid + blockDim.x;

		// compute spacing to avoid bank conflicts
		bankOffsetA = CONFLICT_FREE_OFFSET(ai);
		bankOffsetB = CONFLICT_FREE_OFFSET(bi);

		// Cache the computational window in shared memory
		// pad values beyond n with zeros
		s_data[ai + bankOffsetA] = g_idata[mem_ai]; 
    
		if (isNP2) // compile-time decision
		{
			s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
		}
		else
		{
			s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
		}
	}
	//---------------------------------------------------------------------------
	template <bool isNP2>
	__device__ void storeSharedChunkToMem(int* g_odata, 
										  const int* s_data,
										  int n, 
										  int ai, int bi, 
										  int mem_ai, int mem_bi,
										  int bankOffsetA, int bankOffsetB)
	{
		__syncthreads();

		// write results to global memory
		g_odata[mem_ai] = s_data[ai + bankOffsetA]; 
		if (isNP2) // compile-time decision
		{
			if (bi < n)
				g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
		}
		else
		{
			g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
		}
	}
	//---------------------------------------------------------------------------
	template <bool storeSum>
	__device__ void clearLastElement(int* s_data, 
									 int *g_blockSums, 
									 int blockIndex)
	{
		if (threadIdx.x == 0)
		{
			int index = (blockDim.x << 1) - 1;
			index += CONFLICT_FREE_OFFSET(index);
        
			if (storeSum) // compile-time decision
			{
				// write this block's total sum to the corresponding index in the blockSums array
				g_blockSums[blockIndex] = s_data[index];
			}

			// zero the last element in the scan so it will propagate back to the front
			s_data[index] = 0;
		}
	}
	//---------------------------------------------------------------------------
	__device__ unsigned int buildSum(int *s_data)
	{
		unsigned int thid = threadIdx.x;
		unsigned int stride = 1;
    
		// build the sum in place up the tree
		for (int d = blockDim.x; d > 0; d >>= 1)
		{
			__syncthreads();

			if (thid < d)      
			{
				int i  = __mul24(__mul24(2, stride), thid);
				int ai = i + stride - 1;
				int bi = ai + stride;

				ai += CONFLICT_FREE_OFFSET(ai);
				bi += CONFLICT_FREE_OFFSET(bi);

				s_data[bi] += s_data[ai];
			}

			stride *= 2;
		}

		return stride;
	}
	//---------------------------------------------------------------------------
	__device__ void scanRootToLeaves(int *s_data, unsigned int stride)
	{
		 unsigned int thid = threadIdx.x;

		// traverse down the tree building the scan in place
		for (int d = 1; d <= blockDim.x; d *= 2)
		{
			stride >>= 1;

			__syncthreads();

			if (thid < d)
			{
				int i  = __mul24(__mul24(2, stride), thid);
				int ai = i + stride - 1;
				int bi = ai + stride;

				ai += CONFLICT_FREE_OFFSET(ai);
				bi += CONFLICT_FREE_OFFSET(bi);

				int t  = s_data[ai];
				s_data[ai] = s_data[bi];
				s_data[bi] += t;
			}
		}
	}
	//---------------------------------------------------------------------------
	template <bool storeSum>
	__device__ void prescanBlock(int *data, int blockIndex, int *blockSums)
	{
		int stride = buildSum(data);               // build the sum in place up the tree
		clearLastElement<storeSum>(data, blockSums, 
								   (blockIndex == 0) ? blockIdx.x : blockIndex);
		scanRootToLeaves(data, stride);            // traverse down tree to build the scan 
	}
	//---------------------------------------------------------------------------
	template <bool storeSum, bool isNP2>
	__global__ void prescan(int *g_odata, 
							const int *g_idata, 
							int *g_blockSums, 
							int n, 
							int blockIndex, 
							int baseIndex)
	{
		int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
		extern __shared__ int s_data[];

		// load data into shared memory
		loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, 
									  (baseIndex == 0) ? 
									  __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex,
									  ai, bi, mem_ai, mem_bi, 
									  bankOffsetA, bankOffsetB); 
		// scan the data in each block
		prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
		// write results to device memory
		storeSharedChunkToMem<isNP2>(g_odata, s_data, n, 
									 ai, bi, mem_ai, mem_bi, 
									 bankOffsetA, bankOffsetB);  
	}
	//---------------------------------------------------------------------------
	__global__ void uniformAdd(int *g_data, 
							   int *uniforms, 
							   int n, 
							   int blockOffset, 
							   int baseIndex)
	{
		__shared__ int uni;
		if (threadIdx.x == 0)
			uni = uniforms[blockIdx.x + blockOffset];
    
		unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

		__syncthreads();
    
		// note two adds per thread
		g_data[address]              += uni;
		g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
	}
	//---------------------------------------------------------------------------
	inline bool isPowerOfTwo(int n)
	{
		return ((n&(n-1))==0) ;
	}
	//---------------------------------------------------------------------------
	inline int floorPow2(int n)
	{
	#ifdef WIN32
		// method 2
		return 1 << (int)logb((int)n);
	#else
		// method 1
		// int nf = (int)n;
		// return 1 << (((*(int*)&nf) >> 23) - 127); 
		int exp;
		frexp((int)n, &exp);
		return 1 << (exp - 1);
	#endif
	}
	//---------------------------------------------------------------------------
	int** g_scanBlockSums;
	unsigned int g_numEltsAllocated = 0;
	unsigned int g_numLevelsAllocated = 0;
	//---------------------------------------------------------------------------
	void preallocBlockSums(unsigned int maxNumElements)
	{
		assert(g_numEltsAllocated == 0); // shouldn't be called 

		g_numEltsAllocated = maxNumElements;

		unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
		unsigned int numElts = maxNumElements;

		int level = 0;

		do
		{       
			unsigned int numBlocks = 
				std::max(1, (int)ceil(numElts / (2 * blockSize)));
			if (numBlocks > 1)
			{
				level++;
			}
			numElts = numBlocks;
		} while (numElts > 1);

		g_scanBlockSums = (int**) malloc(level * sizeof(int*));
		g_numLevelsAllocated = level;
    
		numElts = maxNumElements;
		level = 0;
    
		do
		{       
			unsigned int numBlocks = 
				std::max(1, (int)ceil(numElts / (2 * blockSize)));
			if (numBlocks > 1) 
			{
				CUDA_SAFE_CALL(cudaMalloc((void**) &g_scanBlockSums[level++],  
										  numBlocks * sizeof(int)));
			}
			numElts = numBlocks;
		} while (numElts > 1);

		//cutilCheckMsg("preallocBlockSums");
	}
	//---------------------------------------------------------------------------
	void deallocBlockSums()
	{
		for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
		{
			CUDA_SAFE_CALL(cudaFree(g_scanBlockSums[i]));
		}

		//cutilCheckMsg("deallocBlockSums");
    
		free((void**)g_scanBlockSums);

		g_scanBlockSums = 0;
		g_numEltsAllocated = 0;
		g_numLevelsAllocated = 0;
	}
	//---------------------------------------------------------------------------
	void prescanArrayRecursive(int *outArray, 
							   const int *inArray, 
							   int numElements, 
							   int level)
	{
		unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
		unsigned int numBlocks = 
			std::max(1, (int)ceil(numElements / (2 * blockSize)));
		unsigned int numThreads;

		if (numBlocks > 1)
			numThreads = blockSize;
		else if (isPowerOfTwo(numElements))
			numThreads = numElements / 2;
		else
			numThreads = floorPow2(numElements);

		unsigned int numEltsPerBlock = numThreads * 2;

		// if this is a non-power-of-2 array, the last block will be non-full
		// compute the smallest power of 2 able to compute its scan.
		unsigned int numEltsLastBlock = 
			numElements - (numBlocks-1) * numEltsPerBlock;
		unsigned int numThreadsLastBlock = std::max(1, (int)(numEltsLastBlock / 2));
		unsigned int np2LastBlock = 0;
		unsigned int sharedMemLastBlock = 0;
    
		if (numEltsLastBlock != numEltsPerBlock)
		{
			np2LastBlock = 1;

			if(!isPowerOfTwo(numEltsLastBlock))
				numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
			unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
			sharedMemLastBlock = 
				sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
		}

		// padding space is used to avoid shared memory bank conflicts
		unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
		unsigned int sharedMemSize = 
			sizeof(int) * (numEltsPerBlock + extraSpace);

	#ifdef DEBUG
		if (numBlocks > 1)
		{
			assert(g_numEltsAllocated >= numElements);
		}
	#endif

		// setup execution parameters
		// if NP2, we process the last block separately
		dim3  grid(std::max(1, (int)(numBlocks - np2LastBlock)), 1, 1);
		dim3  threads(numThreads, 1, 1);

		// make sure there are no CUDA errors before we start
		//cutilCheckMsg("prescanArrayRecursive before kernels");

		// execute the scan
		if (numBlocks > 1)
		{
			prescan<true, false><<< grid, threads, sharedMemSize >>>(outArray, 
																	 inArray, 
																	 g_scanBlockSums[level],
																	 numThreads * 2, 0, 0);
			CUDA_SAFE_CALL(cudaGetLastError());
			//cutilCheckMsg("prescanWithBlockSums");
			if (np2LastBlock)
			{
				prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
					(outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, 
					 numBlocks - 1, numElements - numEltsLastBlock);
				CUDA_SAFE_CALL(cudaGetLastError());
				//cutilCheckMsg("prescanNP2WithBlockSums");
			}

			// After scanning all the sub-blocks, we are mostly done.  But now we 
			// need to take all of the last values of the sub-blocks and scan those.  
			// This will give us a new value that must be sdded to each block to 
			// get the final results.
			// recursive (CPU) call
			prescanArrayRecursive(g_scanBlockSums[level], 
								  g_scanBlockSums[level], 
								  numBlocks, 
								  level+1);

			uniformAdd<<< grid, threads >>>(outArray, 
											g_scanBlockSums[level], 
											numElements - numEltsLastBlock, 
											0, 0);
			CUDA_SAFE_CALL(cudaGetLastError());
			//cutilCheckMsg("uniformAdd");
			if (np2LastBlock)
			{
				uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, 
														 g_scanBlockSums[level], 
														 numEltsLastBlock, 
														 numBlocks - 1, 
														 numElements - numEltsLastBlock);
				CUDA_SAFE_CALL(cudaGetLastError());
				//cutilCheckMsg("uniformAdd");
			}
		}
		else if (isPowerOfTwo(numElements))
		{
			prescan<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray,
																	  0, numThreads * 2, 0, 0);
			CUDA_SAFE_CALL(cudaGetLastError());
			//cutilCheckMsg("prescan");
		}
		else
		{
			 prescan<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 
																	  0, numElements, 0, 0);
			 CUDA_SAFE_CALL(cudaGetLastError());
			 //cutilCheckMsg("prescanNP2");
		}
	}
	//---------------------------------------------------------------------------
	void prescanArray(int *outArray, int *inArray, int numElements)
	{
		prescanArrayRecursive(outArray, inArray, numElements, 0);
	}
	//---------------------------------------------------------------------------
	int  prefixSum( int* d_inArr, int* d_outArr, int numRecords ) {	
		preallocBlockSums(numRecords);
		prescanArray( d_outArr, d_inArr, numRecords );
		deallocBlockSums();	

		// copy last element
		int* h_outLast = ( int* )malloc( sizeof( int ) );
		CUDA_SAFE_CALL( cudaMemcpy( h_outLast, d_outArr+numRecords-1, sizeof(int), cudaMemcpyDeviceToHost) );
		int* h_inLast = ( int* )malloc( sizeof( int ) );
		CUDA_SAFE_CALL( cudaMemcpy( h_inLast, d_inArr+numRecords-1, sizeof(int), cudaMemcpyDeviceToHost) );

		unsigned int sum = *h_outLast + *h_inLast;

		free( h_outLast );
		free( h_inLast );
	
		return sum;
	}
	//---------------------------------------------------------------------------
}}