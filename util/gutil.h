/* Common Definition */

#ifndef _GPSM_COMMON_H_
#define _GPSM_COMMON_H_

#include "cuda.h"
#include "cutil.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iostream>

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
	    }} while(0)

#define NP2(n)              do {                    \
    n--;                                            \
    n |= n >> 1;                                    \
    n |= n >> 2;                                    \
    n |= n >> 4;                                    \
    n |= n >> 8;                                    \
    n |= n >> 16;                                   \
    n ++; } while (0) 

#define FOR_LIMIT(i, len)			for(int i=0; i<len; i++)

#define FOR_LIMIT_REV(i, len)		for(int i=len-1; i>=0; i--)

#define FOR_RANGE(i, begin, end)		for(int i=begin; i<end; i++)

#define FOR_RANGE_REV(i, begin, end)	for(int i=end-1; i>=begin; i--)

#define FILL(arr, size, val)	for(int i=0; i< size; i++) arr[i] = val

#define WARP_SIZE (32)
#define BLOCK_SIZE (512)

#define THREADS_PER_BLOCK (blockDim.x)
#define WARPS_PER_BLOCK ((THREADS_PER_BLOCK-1)/WARP_SIZE + 1)
#define BLOCKS_PER_GRID (gridDim.x)

#define BID	(blockIdx.x)
#define TID	(threadIdx.x)	/*thread ID in current block*/
#define WID (TID / WARP_SIZE)
#define WTID (TID % WARP_SIZE)

#define GTID (BID * THREADS_PER_BLOCK + TID)	/*global thread ID*/
#define GWID (GTID / WARP_SIZE)
#define GWTID (GTID % WARP_SIZE)

#define TOTAL_THREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

#ifdef __INTELLISENSE__
#define __launch_bounds__(a,b)
void __syncthreads(void);
void __threadfence(void);
int __mul24(int, int);
#endif

typedef unsigned int uint;

namespace gpsm {

	enum DataPosition { /* data position */
		GPU,
		MEM,
		PINNED,
		MMAP,
		DISK
	};

	enum CopyType { /* supported copy types */
		HOST_TO_DEVICE = 0,
		HOST_TO_HOST,
		DEVICE_TO_HOST
	};

	enum Direction { /* scanning directions */
		IN,
		OUT
	};


	struct GPPlan {
		int numNodes;
		int* nodes;
		bool* scanIn;
		bool* scanOut;
		
		GPPlan() {
			numNodes = 0;
			nodes = NULL;
			scanIn = NULL;
			scanOut = NULL;
		}

		~GPPlan() {
			if (numNodes > 0) {
				free(nodes);
				free(scanIn);
				free(scanOut);
			}
		}
	};

	struct GPSpec{

	};
}

#endif