#ifndef __GPSM_EXCEPTION_NODE_H__
#define __GPSM_EXCEPTION_NODE_H__

#include "gutil.h"
#include "graph.h"

namespace gpsm {
namespace graph {
	
	struct GPExpNode
	{
		int threshole;
		int labelCount;
		int* highInDegCounts;
		int** highInDegNodes;

		int* highOutDegCounts;
		int** highOutDegNodes;
		bool inDev;

		GPExpNode(int t = BLOCK_SIZE){
			threshole = t;
			labelCount = 0;
			highInDegCounts = NULL;
			highInDegNodes = NULL;

			highOutDegCounts = NULL;
			highOutDegNodes = NULL;
			inDev = false;
		}

		~GPExpNode() {
			if (labelCount > 0) {
				if (inDev) {
					FOR_LIMIT(i, labelCount) {
						if (highInDegCounts[i] > 0) CUDA_SAFE_CALL(cudaFree(highInDegNodes[i]));
						if (highOutDegCounts[i] > 0) CUDA_SAFE_CALL(cudaFree(highOutDegNodes[i]));
					}
				}
				else {
					FOR_LIMIT(i, labelCount) {
						if (highInDegCounts[i] > 0) delete[] highInDegNodes[i];
						if (highOutDegCounts[i] > 0) delete[] highOutDegNodes[i];
					}
				}

				delete[] highInDegCounts;
				delete[] highOutDegCounts;

				delete[] highInDegNodes;
				delete[] highOutDegNodes;
			}
		}

		/* Extract high-degree nodes from a graph */
		void extract(GPGraph* graph) {
			labelCount = graph->numLabels;
			
			highInDegCounts = new int[labelCount];
			CHECK_POINTER(highInDegCounts);
			FILL(highInDegCounts, labelCount, 0);

			highInDegNodes = new int*[labelCount];
			CHECK_POINTER(highInDegNodes);

			highOutDegCounts = new int[labelCount];
			CHECK_POINTER(highOutDegCounts);
			FILL(highOutDegCounts, labelCount, 0);

			highOutDegNodes = new int*[labelCount];
			CHECK_POINTER(highOutDegNodes);

			FOR_LIMIT(i, graph->numNodes) {
				int inDegree = graph->inOffsets[i + 1] - graph->inOffsets[i];
				int outDegree = graph->outOffsets[i + 1] - graph->outOffsets[i];
				int label = graph->nodeLabels[i];

				if (inDegree > threshole) highInDegCounts[label]++;
				if (outDegree > threshole) highOutDegCounts[label]++;
			}

			FOR_LIMIT(i, labelCount) {
				if (highInDegCounts[i] > 0) highInDegNodes[i] = new int[highInDegCounts[i]];
				if (highOutDegCounts[i] > 0) highOutDegNodes[i] = new int[highOutDegCounts[i]];
			}

			FILL(highInDegCounts, labelCount, 0);
			FILL(highOutDegCounts, labelCount, 0);

			FOR_LIMIT(i, graph->numNodes) {
				int inDegree = graph->inOffsets[i + 1] - graph->inOffsets[i];
				int outDegree = graph->outOffsets[i + 1] - graph->outOffsets[i];
				int label = graph->nodeLabels[i];

				if (inDegree > threshole) highInDegNodes[label][highInDegCounts[label]++] = i;

				if (outDegree > threshole) highOutDegNodes[label][highOutDegCounts[label]++] = i;
			}
		}

		/* Copy */
		GPExpNode* copy(CopyType type) {
			if ((type == HOST_TO_DEVICE || type == HOST_TO_HOST)
				&& inDev == true) return NULL;

			if (type == DEVICE_TO_HOST && inDev == false) return NULL;

			GPExpNode* dest = new GPExpNode();

			dest->labelCount = labelCount;
			dest->threshole = threshole;

			dest->highInDegCounts = new int[labelCount];
			CHECK_POINTER(dest->highInDegCounts);

			dest->highInDegNodes = new int*[labelCount];
			CHECK_POINTER(highInDegNodes);

			dest->highOutDegCounts = new int[labelCount];
			CHECK_POINTER(dest->highOutDegCounts);

			dest->highOutDegNodes = new int*[labelCount];
			CHECK_POINTER(dest->highOutDegNodes);

			memcpy(dest->highInDegCounts, highInDegCounts, labelCount * sizeof(int));
			memcpy(dest->highOutDegCounts, highOutDegCounts, labelCount * sizeof(int));

			switch (type)
			{
			case HOST_TO_DEVICE:
				dest->inDev = true;
				FOR_LIMIT(i, labelCount) {
					if (dest->highInDegCounts[i] > 0) {
						CUDA_SAFE_CALL(cudaMalloc(&dest->highInDegNodes[i], dest->highInDegCounts[i] * sizeof(int)));
						CUDA_SAFE_CALL(cudaMemcpy(dest->highInDegNodes[i], highInDegNodes[i], dest->highInDegCounts[i] * sizeof(int),
							cudaMemcpyHostToDevice));
					}

					if (dest->highOutDegCounts[i] > 0) {
						CUDA_SAFE_CALL(cudaMalloc(&dest->highOutDegNodes[i], dest->highOutDegCounts[i] * sizeof(int)));
						CUDA_SAFE_CALL(cudaMemcpy(dest->highOutDegNodes[i], highOutDegNodes[i], dest->highOutDegCounts[i] * sizeof(int),
							cudaMemcpyHostToDevice));
					}
				}
				break;
			case DEVICE_TO_HOST:
				dest->inDev = false;
				FOR_LIMIT(i, labelCount) {
					if (dest->highInDegCounts[i] > 0) {
						dest->highInDegNodes[i] = new int[dest->highInDegCounts[i]];
						CUDA_SAFE_CALL(cudaMemcpy(dest->highInDegNodes[i], highInDegNodes[i], dest->highInDegCounts[i] * sizeof(int),
							cudaMemcpyDeviceToHost));
					}

					if (dest->highOutDegCounts[i] > 0) {
						dest->highOutDegNodes[i] = new int[dest->highOutDegCounts[i]];
						CUDA_SAFE_CALL(cudaMemcpy(dest->highOutDegNodes[i], highOutDegNodes[i], dest->highOutDegCounts[i] * sizeof(int),
							cudaMemcpyDeviceToHost));
					}
				}
				break;
			case HOST_TO_HOST:
				dest->inDev = false;
				FOR_LIMIT(i, labelCount) {
					if (dest->highInDegCounts[i] > 0) {
						dest->highInDegNodes[i] = new int[dest->highInDegCounts[i]];
						memcpy(dest->highInDegNodes[i], highInDegNodes[i], dest->highInDegCounts[i] * sizeof(int));
					}

					if (dest->highOutDegCounts[i] > 0) {
						dest->highOutDegNodes[i] = new int[dest->highOutDegCounts[i]];
						memcpy(dest->highOutDegNodes[i], highOutDegNodes[i], dest->highOutDegCounts[i] * sizeof(int));
					}
				}
			default:
				break;
			}

			return dest;
		}
	};
}}

#endif
