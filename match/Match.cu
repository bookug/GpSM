/*=============================================================================
# Filename: Match.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-12-15 01:38
# Description: 
=============================================================================*/

#include "Match.h"

using namespace std;

void 
Match::initGPU(int dev)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(dev);
	//NOTE: 48KB shared memory per block, 1024 threads per block, 30 SMs and 128 cores per SM
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %luB; compute v%d.%d; clock: %d kHz; shared mem: %dB; block threads: %d; SM count: %d\n",
               devProps.name, devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate,
			   devProps.sharedMemPerBlock, devProps.maxThreadsPerBlock, devProps.multiProcessorCount);
    }
	cout<<"GPU selected"<<endl;
	//GPU initialization needs several seconds, so we do it first and only once
	//https://devtalk.nvidia.com/default/topic/392429/first-cudamalloc-takes-long-time-/
	int* warmup = NULL;
	/*unsigned long bigg = 0x7fffffff;*/
	/*cudaMalloc(&warmup, bigg);*/
	/*cout<<"warmup malloc"<<endl;*/
	cudaMalloc(&warmup, sizeof(int));
	cudaFree(warmup);
	cout<<"GPU warmup finished"<<endl;
	size_t size = 0x7fffffff;
	/*size *= 3;   //heap corruption for 3 and 4*/
	size *= 2;
	/*size *= 2;*/
	//NOTICE: the memory alloced by cudaMalloc is different from the GPU heap(for new/malloc in kernel functions)
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	cout<<"check heap limit: "<<size<<endl;

	// Runtime API
	// cudaFuncCachePreferShared: shared memory is 48 KB
	// cudaFuncCachePreferEqual: shared memory is 32 KB
	// cudaFuncCachePreferL1: shared memory is 16 KB
	// cudaFuncCachePreferNone: no preference
	/*cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferShared)*/
	//The initial configuration is 48 KB of shared memory and 16 KB of L1 cache
	//The maximum L2 cache size is 3 MB.
	//also 48 KB read-only cache: if accessed via texture/surface memory, also called texture cache;
	//or use _ldg() or const __restrict__
	//64KB constant memory, ? KB texture memory. cache size?
	//CPU的L1 cache是根据时间和空间局部性做出的优化，但是GPU的L1仅仅被设计成针对空间局部性而不包括时间局部性。频繁的获取L1不会导致某些数据驻留在cache中，只要下次用不到，直接删。
	//L1 cache line 128B, L2 cache line 32B, notice that load is cached while store not
	//mmeory read/write is in unit of a cache line
	//the word size of GPU is 32 bits
}

Match::Match(Graph* _query, Graph* _data)
{
	this->query = _query;
	this->data = _data;
	id2pos = pos2id = NULL;
}

Match::~Match()
{
	delete[] this->id2pos;
}

inline void 
Match::add_mapping(int _id)
{
	pos2id[current_pos] = _id;
	id2pos[_id] = current_pos;
	this->current_pos++;
}

/*bool is_start = true;*/
/*int*/
/*Match::get_minimum_idx(float* score, int qsize)*/
/*{*/
	/*float* min_ptr = min_element(score, score+qsize);*/
	/*[>if(is_start)<]*/
	/*[>{<]*/
		/*[>is_start = false;<]*/
		/*[>min_ptr = score + 4;<]*/
	/*[>}<]*/
	/*int min_idx = min_ptr - score;*/
	/*memset(min_ptr, 0x7f, sizeof(float));*/
	/*[>thrust::device_ptr<float> dev_ptr(d_score);<]*/
	/*[>float* min_ptr = thrust::raw_pointer_cast(thrust::min_element(dev_ptr, dev_ptr+qsize));<]*/
	/*[>int min_idx = min_ptr - d_score;<]*/
	/*[>//set this node's score to maximum so it won't be chosed again<]*/
	/*[>cudaMemset(min_ptr, 0x7f, sizeof(float));<]*/

	/*//NOTICE: memset is used per-byte, so do not set too large value, otherwise it will be negative*/
	/*//http://blog.csdn.net/Vmurder/article/details/46537613*/
	/*[>cudaMemset(d_score+min_idx, 1000.0f, sizeof(float));<]*/
	/*[>float tmp = 0.0f;<]*/
	/*[>cout<<"to check the score: ";<]*/
	/*[>for(int i = 0; i < qsize; ++i)<]*/
	/*[>{<]*/
		/*[>cudaMemcpy(&tmp, d_score+i, sizeof(float), cudaMemcpyDeviceToHost);<]*/
		/*[>cout<<tmp<<" ";<]*/
	/*[>}cout<<endl;<]*/
/*#ifdef DEBUG*/
	/*checkCudaErrors(cudaGetLastError());*/
/*#endif*/

	/*this->add_mapping(min_idx);*/
	/*return min_idx;*/
/*}*/

void
Match::copyGraphToGPU()
{
	cudaMalloc(&d_data_row_offset, sizeof(unsigned)*(this->data->vertex_num+1));
	cudaMemcpy(d_data_row_offset, this->data->undirected_row_offset, sizeof(unsigned)*(this->data->vertex_num+1), cudaMemcpyHostToDevice);
	int edge_num = this->data->eSize();
	cudaMalloc(&d_data_column_index, sizeof(unsigned)*edge_num);
	cudaMemcpy(d_data_column_index, this->data->undirected_column_index, sizeof(unsigned)*edge_num, cudaMemcpyHostToDevice);
	cudaMalloc(&d_data_vertex_value, sizeof(unsigned)*(this->data->vertex_num));
	cudaMemcpy(d_data_vertex_value, this->data->vertex_value, sizeof(unsigned)*(this->data->vertex_num), cudaMemcpyHostToDevice);

#ifdef DEBUG
	/*cout<<"data graph already in GPU"<<endl;*/
	checkCudaErrors(cudaGetLastError());
#endif
}

__host__ unsigned
binary_search_cpu(unsigned _key, unsigned* _array, unsigned _array_num)
{
	//MAYBE: return the right border, or use the left border and the right border as parameters
    if (_array_num == 0 || _array == NULL)
    {
		return INVALID;
    }

    unsigned _first = _array[0];
    unsigned _last = _array[_array_num - 1];

    if (_last == _key)
    {
        return _array_num - 1;
    }

    if (_last < _key || _first > _key)
    {
		return INVALID;
    }

    unsigned low = 0;
    unsigned high = _array_num - 1;
    unsigned mid;
    while (low <= high)
    {
        mid = (high - low) / 2 + low;   //same to (low+high)/2
        if (_array[mid] == _key)
        {
            return mid;
        }
        if (_array[mid] > _key)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
	return INVALID;
}

//BETTER: maybe we can use dynamic parallism here
__device__ unsigned
binary_search(unsigned _key, unsigned* _array, unsigned _array_num)
{
//http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#host
/*#if defined(__CUDA_ARCH__)*/
	//MAYBE: return the right border, or use the left border and the right border as parameters
    if (_array_num == 0 || _array == NULL)
    {
		return INVALID;
    }

    unsigned _first = _array[0];
    unsigned _last = _array[_array_num - 1];

    if (_last == _key)
    {
        return _array_num - 1;
    }

    if (_last < _key || _first > _key)
    {
		return INVALID;
    }

    unsigned low = 0;
    unsigned high = _array_num - 1;
    unsigned mid;
    while (low <= high)
    {
        mid = (high - low) / 2 + low;   //same to (low+high)/2
        if (_array[mid] == _key)
        {
            return mid;
        }
        if (_array[mid] > _key)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
	return INVALID;
/*#else*/
/*#endif*/
}

__device__ void
get_neighbors(unsigned id, int label, unsigned*& list, unsigned& list_num, unsigned* d_data_row_offset, unsigned* d_data_edge_value, unsigned* d_data_edge_offset,  unsigned* d_data_column_index)
{
	/*printf("thread %d: to get neighbors for id %d\n", threadIdx.x, id);*/
	/*int tmp = d_data_column_index[1];*/
	/*printf("thread %d check neighbor: %d\n", threadIdx.x, tmp);*/
	unsigned start = d_data_row_offset[id], end = d_data_row_offset[id+1];
	unsigned i = binary_search(label, d_data_edge_value+start, end-start);
	if(i == INVALID)  // not found
	{
		list = NULL;
		list_num = 0;
		return;
	}
	i += start;
	start = d_data_edge_offset[i];
	end = d_data_edge_offset[i+1];
	list_num = end - start;
	list = d_data_column_index + start;
}

//int *result = new int[1000];
/*int *result_end = thrust::set_intersection(A1, A1 + size1, A2, A2 + size2, result, thrust::less<int>());*/
//
//BETTER: choose between merge-join and bianry-search, or using multiple threads to do intersection
//or do inetrsection per-element, use compact operation finally to remove invalid elements
__device__ void
intersect(unsigned*& cand, unsigned& cand_num, unsigned* list, unsigned list_num)
{
	int i, j, cnt = 0;
	for(i = 0; i < cand_num; ++i)
	{
		unsigned found = binary_search(cand[i], list, list_num);
		if(found == INVALID)
		{
			cand[i] = INVALID;
		}
		else
		{
			cnt++;
		}
	}

	if(cnt == 0)
	{
		delete[] cand;
		cand = NULL;
	}
	else
	{
		for(i = 0, j = 0; i < cand_num; ++i)
		{
			if(cand[i] != INVALID)
			{
				cand[j++] = cand[i];
			}
		}
	}
	cand_num = cnt;
}

__device__ void
subtract(unsigned*& cand, unsigned& cand_num, unsigned* record, unsigned result_col_num)
{
	int i, j, cnt = cand_num;
	//BETTER?: how to improve to avoid new memory? use merge-join
	bool* invalid = new bool[cand_num];
	memset(invalid, false, sizeof(bool)*cand_num);
	for(i = 0; i < result_col_num; ++i)
	{
		//NOTICE: there is no duplicates in the candidate set
		unsigned found = binary_search(record[i], cand, cand_num);
		if(found != INVALID)
		{
			cnt--;
			//WARN: this will cause error bercause next iteration there may be INVALID in this array!
			/*cand[found] = INVALID;*/
			invalid[found] = true;
		}
	}

	if(cnt > 0)
	{
		for(i = 0, j = 0; i < cand_num; ++i)
		{
			if(!invalid[i])
			{
				cand[j++] = cand[i];
			}
		}
	}
	else
	{
		delete[] cand;
		cand = NULL;
	}
	delete[] invalid;
	cand_num = cnt;
}

void 
Match::generateSpanningTree(Graph* span_tree, unsigned* filter_order)
{
	unsigned* inverse_label = this->data->inverse_label;
	unsigned* inverse_num = this->data->inverse_num;
	unsigned label_num = this->data->label_num;
	int qsize = this->query->vertex_num;
	float* score = new float[qsize];
	
	for(int i = 0; i < qsize; ++i)
	{
		float num = 0.0f;
		unsigned label = this->query->vertex_value[i];
		for(int j = 0; j < label_num; ++j)
		{
			if(inverse_label[j] == label)
			{
				num = inverse_num[j];
				break;
			}
		}
		int degree = this->query->vertices[i].eSize();
		score[i] = (degree+0.0)/num;
	}

	/*cout<<"check score: "<<endl;*/
	/*for(int i = 0; i < qsize; ++i)*/
	/*{*/
		/*cout<<score[i]<<" ";*/
	/*}cout<<endl;*/

	float maxv = 0.0f;
	int maxi = -1, from, to;
	/*cout<<"edge num: "<<edge_num<<endl;*/
	for(int i = 0; i < edge_num; ++i)
	{
		from = this->edge_from[i], to = this->edge_to[i];
		/*cout<<"check: "<<from<<" "<<to<<" "<<score[from]<<" "<<score[to]<<endl;*/
		float tmp = score[from] + score[to];
		if(tmp > maxv)
		{
			maxv = tmp;
			maxi = i;
		}
	}
	from = this->edge_from[maxi];
	to = this->edge_to[maxi];
	int start = from;
	if(score[to] > score[from])
	{
		start = to;
	}

	bool* visit = new bool[qsize];
	memset(visit, false, sizeof(bool)*qsize);
	unsigned* mapping = new unsigned[qsize];
	int vid = 0;
	//construct the spanning tree, the label is the real ID in the query graph
	filter_order[vid] = start;
	span_tree->addVertex(start);
	visit[start] = true;
	mapping[start] = vid;
	vid++;

	queue<unsigned> tq;
	tq.push(start);
	while(!tq.empty())
	{
		int tmp = tq.front();
		tq.pop();
		for(int i = this->query->undirected_row_offset[tmp]; i < this->query->undirected_row_offset[tmp+1]; ++i)
		{
			int adj = this->query->undirected_column_index[i];
			if(visit[adj])
			{
				continue;
			}
			tq.push(adj);
			filter_order[vid] = adj;
			span_tree->addVertex(adj);
			//we do not care about the label here
			span_tree->addEdge(vid, mapping[tmp], 1);
			visit[adj] = true;
			mapping[adj] = vid;
			vid++;
		}
	}

	delete[] score;
	delete[] visit;
	delete[] mapping;
}

__global__ void
check_kernel(unsigned* d_data_vertex_value, unsigned* d_data_row_offset, bool* d_valid, int dsize, int label, int degree)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= dsize)
	{
		return; 
	}
	int data_label = d_data_vertex_value[i];
	//BETTER: overlapping exists, may use shared memory to optimize
	int data_degree = d_data_row_offset[i+1] - d_data_row_offset[i];
	if(data_label == label && data_degree >= degree)
	{
		d_valid[i] = true;
	}
}

void 
Match::kernel_check(int u, bool* d_valid)
{
	int dsize = this->data->vertex_num;
	int label = this->query->vertex_value[u];
	int degree = this->query->getVertexDegree(u);
	/*cout<<"check kernel: "<<u<<" "<<label<<" "<<degree<<endl;*/
	int BLOCK_SIZE = 1024;
	int GRID_SIZE = (dsize+BLOCK_SIZE-1)/BLOCK_SIZE;
	check_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_data_vertex_value, d_data_row_offset, d_valid, dsize, label, degree);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}
	
__global__ void
transform_kernel(bool* d_valid, unsigned* d_offset, int dsize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= dsize)
	{
		return; 
	}
	d_offset[i] = (d_valid[i])?1:0;
}

__global__ void
scatter_kernel(bool* d_valid, int dsize, unsigned* d_offset, unsigned* d_array)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= dsize)
	{
		return; 
	}
	if(d_valid[i])
	{
		d_array[d_offset[i]] = i;
	}
}

void
Match::kernel_collect(bool* d_valid, unsigned*& d_array, unsigned& d_array_num)
{
	/*thrust::device_ptr<unsigned> dev_ptr(X.size());*/
	/*thrust::copy_if(in_array, in_array + size, out_array, is_not_zero);*/
	/*thrust::remove_if(in_array, in_array + size, is_zero);*/
	int dsize = this->data->vertex_num;

	//compact operation
	unsigned* d_offset = NULL;
	checkCudaErrors(cudaMalloc(&d_offset, sizeof(unsigned)*(dsize+1)));
	/*cudaMemset(d_offset, 0, sizeof(unsigned)*(dsize+1));*/
	/*thrust::transform(d_valid, d_valid+dsize, d_offset, d_offset+dsize, bool2uint_functor());*/
	transform_kernel<<<(dsize+1023)/1024,1024>>>(d_valid, d_offset, dsize);
	cudaDeviceSynchronize();

	thrust::device_ptr<unsigned> dev_ptr(d_offset);
	int sum;
	thrust::exclusive_scan(dev_ptr, dev_ptr+dsize+1, dev_ptr);
	checkCudaErrors(cudaGetLastError());
	cudaMemcpy(&sum, d_offset+dsize, sizeof(unsigned), cudaMemcpyDeviceToHost);

	if(sum == 0)
	{
		return;
	}

	d_array_num = sum;
	checkCudaErrors(cudaMalloc(&d_array, sizeof(unsigned)*sum));
	int BLOCK_SIZE = 1024;
	int GRID_SIZE = (dsize+BLOCK_SIZE-1)/BLOCK_SIZE;
	scatter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_valid, dsize, d_offset, d_array);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_offset));
	checkCudaErrors(cudaGetLastError());
}

__global__ void
explore_kernel(unsigned* d_data_vertex_value, unsigned* d_data_row_offset, unsigned* d_data_column_index, bool* d_candidate_set, int dsize, unsigned* d_array, unsigned d_array_num, unsigned u, unsigned* d_query_adj_u, unsigned query_adj_u_num)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int group = t / 32;
	int idx = t % 32;
	if(group >= d_array_num)
	{
		return; 
	}
	unsigned mu = d_array[group];
	/*if(mu == 1 && idx==0)*/
	/*{*/
		/*printf("found mu 1 and idx 0\n");*/
	/*}*/
	unsigned begin = d_data_row_offset[mu];
	unsigned size = d_data_row_offset[mu+1] - begin;
	unsigned loop = size / 32;
	size = size % 32;
	for(int i = 0; i < query_adj_u_num; ++i)
	{
		bool invalid = false;
		int v = d_query_adj_u[i];
		int label = d_query_adj_u[i+query_adj_u_num];
		int degree = d_query_adj_u[i+2*query_adj_u_num];
		/*if(u == 1 && mu == 1 && v == 2)*/
		/*{*/
			/*printf("check label and degree: %d %d\n", label, degree);*/
		/*}*/
		//each thread get adjacent vertex of mu as mv
		unsigned base = 0;
		for(int j = 0; j < loop; ++j, base+=32)
		{
			unsigned mv = d_data_column_index[begin+idx+base];
			unsigned data_label = d_data_vertex_value[mv];
			unsigned data_degree = d_data_row_offset[mv+1] - d_data_row_offset[mv];
			if(data_label == label && data_degree >= degree)
			{
				d_candidate_set[dsize*v+mv] = true;
				invalid = true;
			}
		}
		if(idx < size)
		{
			unsigned mv = d_data_column_index[begin+idx+base];
			unsigned data_label = d_data_vertex_value[mv];
			unsigned data_degree = d_data_row_offset[mv+1] - d_data_row_offset[mv];
			/*printf("to check matching %d %d %d\n", u, mu, v);*/
		/*if(u == 1 && mu == 2 && v == 2)*/
		/*{*/
			/*printf("check matching: %d %d %d %d\n", idx, mv, data_label, data_degree);*/
			/*printf("loop and base: %d %d %d\n", loop, base, size);*/
		/*}*/
			if(data_label == label && data_degree >= degree)
			{
		/*if(u == 1 && mu == 1 && v == 2)*/
		/*{*/
			/*printf("successful matching: %d %d %d %d\n", idx, mv, data_label, data_degree);*/
		/*}*/
				d_candidate_set[dsize*v+mv] = true;
				invalid = true;
			}
		}
		//NOTICE: invalid is a private variable for each thread
		//USAGE: warp vote functions   __all()   __any(int predicate)  __ballot()
		//REFERENCE: https://stackoverflow.com/questions/10557254/about-warp-voting-function
		if(__any(invalid) == 0)
		{
			/*printf("return u, mu, v: %d %d %d %d\n", u, mu, v);*/
			d_candidate_set[dsize*u+mu] = false;
			return;
		}
	}
}

void
Match::kernel_explore(int u, unsigned* d_array, unsigned d_array_num, bool* d_candidate_set, Graph* span_tree, bool* initialized)
{
	int dsize = this->data->vertex_num;
	unsigned query_adj_u_num = this->query->getVertexDegree(u);
	//NOTICE: no need to explore vertices which are initialized already
	for(int i = this->query->undirected_row_offset[u]; i < this->query->undirected_row_offset[u+1]; ++i)
	{
		unsigned adj = this->query->undirected_column_index[i];
		if(initialized[adj])
		{
			query_adj_u_num--;
		}
	}
	if(query_adj_u_num == 0)
	{
		return; 
	}

	unsigned* d_query_adj_u = NULL;
	checkCudaErrors(cudaMalloc(&d_query_adj_u, sizeof(unsigned)*query_adj_u_num*3));
	unsigned* h_query_adj_u = new unsigned[query_adj_u_num * 3];
	/*memcpy(h_query_adj_u, this->query->undirected_column_index+this->query->undirected_row_offset[u], sizeof(unsigned)*query_adj_u_num);*/
	int j = 0;
	for(int i = this->query->undirected_row_offset[u]; i < this->query->undirected_row_offset[u+1]; ++i)
	{
		unsigned adj = this->query->undirected_column_index[i];
		if(!initialized[adj])
		{
			h_query_adj_u[j++] = adj;
		}
	}

	for(int i = 0; i < query_adj_u_num; ++i)
	{
		h_query_adj_u[i+query_adj_u_num] = this->query->vertex_value[h_query_adj_u[i]];
		h_query_adj_u[i+2*query_adj_u_num] = this->query->getVertexDegree(h_query_adj_u[i]);
		/*cout<<"double check: "<<h_query_adj_u[i]<<" "<<h_query_adj_u[i+query_adj_u_num]<<" "<<h_query_adj_u[i+2*query_adj_u_num]<<endl;*/
	}
	cudaMemcpy(d_query_adj_u, h_query_adj_u, sizeof(unsigned)*query_adj_u_num*3, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	delete[] h_query_adj_u;

	//assign a warp to each candidate vertex
	int WARP_SIZE = 32;
	int BLOCK_SIZE = 256;
	/*int BLOCK_SIZE = 64;*/
	int GRID_SIZE = (d_array_num*WARP_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
	/*cout<<"check array num: "<<u<<" "<<d_array_num<<" "<<GRID_SIZE<<" "<<WARP_SIZE<<endl;*/
	explore_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_data_vertex_value, d_data_row_offset, d_data_column_index, d_candidate_set, dsize, d_array, d_array_num, u, d_query_adj_u, query_adj_u_num);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_query_adj_u));
	cudaDeviceSynchronize();
}

bool
Match::InitializeCandidateVertices(Graph* span_tree, unsigned* filter_order, bool* d_candidate_set)
{
	int qsize = this->query->vertex_num;
	int dsize = this->data->vertex_num;
	bool* initialized = new bool[qsize];
	memset(initialized, false, sizeof(bool)*qsize);
	cudaMemset(d_candidate_set, false, sizeof(bool)*qsize*dsize);

	for(int i = 0; i < qsize; ++i)
	{
		unsigned u = filter_order[i];
		if(!initialized[u])
		{
			/*cout<<"to find candidate for "<<u<<endl;*/
			kernel_check(u, d_candidate_set+dsize*u);
			initialized[u] = true;
		}
		unsigned* d_array = NULL;
		unsigned d_array_num = 0;
		kernel_collect(d_candidate_set+dsize*u, d_array, d_array_num);
		/*cout<<"check cand: "<<u<<" "<<d_array_num<<endl;*/
		if(d_array_num == 0)
		{
			cout<<"no candidate for the "<<u<<"th query vertex"<<endl;
			return false; 
		}
		kernel_explore(u, d_array, d_array_num, d_candidate_set, span_tree, initialized);
		checkCudaErrors(cudaFree(d_array));
		for(int j = this->query->undirected_row_offset[u]; j < this->query->undirected_row_offset[u+1]; ++j)
		{
			int adj = this->query->undirected_column_index[j];
			/*cout<<"already init: "<<u<<" "<<adj<<endl;*/
			initialized[adj] = true;
		}	
	}

	delete[] initialized;
	return true;
}

void 
Match::generateSimplifiedGraph(std::vector<unsigned>& simple_vertices, std::vector< std::vector<unsigned> >& simple_edges, int degree_threshold)
{
	int qsize = this->query->vertex_num;
	bool* valid = new bool[qsize];
	memset(valid, false, sizeof(bool)*qsize);
	for(int i = 0; i < qsize; ++i)
	{
		if(this->query->getVertexDegree(i) > degree_threshold)
		{
			valid[i] = true;
			simple_vertices.push_back(i);
		}
	}
	int simple_num = simple_vertices.size();
	for(int i = 0; i < simple_num; ++i)
	{
		simple_edges.push_back(vector<unsigned>());
		vector<Neighbor>& in = this->query->vertices[simple_vertices[i]].in;
		vector<Neighbor>& out = this->query->vertices[simple_vertices[i]].out;
		for(int j = 0; j < in.size(); ++j)
		{
			int id = in[j].vid;
			if(valid[id])
			{
				simple_edges[i].push_back(id);
			}
		}
		for(int j = 0; j < out.size(); ++j)
		{
			int id = out[j].vid;
			if(valid[id])
			{
				simple_edges[i].push_back(id);
			}
		}
	}
	delete[] valid;
}

__global__ void
refine_kernel(unsigned* d_data_row_offset, unsigned* d_data_column_index, bool* d_candidate_set, int dsize, unsigned* d_array, unsigned d_array_num, unsigned u, unsigned* d_query_adj_u, unsigned query_adj_u_num)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int group = t / 32;
	int idx = t % 32;
	if(group >= d_array_num)
	{
		return; 
	}
	unsigned mu = d_array[group];
	/*if(mu == 1 && idx==0)*/
	/*{*/
		/*printf("found mu 1 and idx 0\n");*/
	/*}*/
	unsigned begin = d_data_row_offset[mu];
	unsigned size = d_data_row_offset[mu+1] - begin;
	unsigned loop = size / 32;
	size = size % 32;
	for(int i = 0; i < query_adj_u_num; ++i)
	{
		bool invalid = false;
		int v = d_query_adj_u[i];
		//each thread get adjacent vertex of mu as mv
		unsigned base = 0;
		for(int j = 0; j < loop; ++j, base+=32)
		{
			unsigned mv = d_data_column_index[begin+idx+base];
			if(d_candidate_set[dsize*v+mv])
			{
				invalid = true;
			}
		}
		if(idx < size)
		{
			unsigned mv = d_data_column_index[begin+idx+base];
			if(d_candidate_set[dsize*v+mv])
			{
				invalid = true;
			}
		}
		if(__any(invalid) == 0)
		{
			d_candidate_set[dsize*u+mu] = false;
			return;
		}
	}
}

void 
Match::kernel_refine(int idx, unsigned* d_array, unsigned d_array_num, bool* d_candidate_set, std::vector<unsigned>& simple_vertices, std::vector< std::vector<unsigned> >& simple_edges)
{
	unsigned u = simple_vertices[idx];
	int dsize = this->data->vertex_num;
	unsigned query_adj_u_num = simple_edges[idx].size();
	if(query_adj_u_num == 0)
	{
		return; 
	}

	unsigned* d_query_adj_u = NULL;
	checkCudaErrors(cudaMalloc(&d_query_adj_u, sizeof(unsigned)*query_adj_u_num));
	unsigned* h_query_adj_u = new unsigned[query_adj_u_num];
	for(int i = 0; i < query_adj_u_num; ++i)
	{
		unsigned adj = simple_edges[idx][i];
		h_query_adj_u[i] = adj;
	}
	cudaMemcpy(d_query_adj_u, h_query_adj_u, sizeof(unsigned)*query_adj_u_num, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	delete[] h_query_adj_u;

	//assign a warp to each candidate vertex
	int WARP_SIZE = 32;
	int BLOCK_SIZE = 256;
	/*int BLOCK_SIZE = 64;*/
	int GRID_SIZE = (d_array_num*WARP_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
	/*cout<<"check array num: "<<u<<" "<<d_array_num<<" "<<GRID_SIZE<<" "<<WARP_SIZE<<endl;*/
	refine_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_data_row_offset, d_data_column_index, d_candidate_set, dsize, d_array, d_array_num, u, d_query_adj_u, query_adj_u_num);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_query_adj_u));
	cudaDeviceSynchronize();
}

bool
Match::RefineCandidateVertices(bool* d_candidate_set)
{
	//BETTER: for large graphs, this threshold should be improved
	int degree_threshold = 1;
	vector<unsigned> simple_vertices;
	vector< vector<unsigned> > simple_edges;
	generateSimplifiedGraph(simple_vertices, simple_edges, degree_threshold);

	int dsize = this->data->vertex_num;
	int simple_num = simple_vertices.size();
	/*cout<<"simple graph: "<<simple_num<<endl;*/

	//NOTICE: theoretically this refining process should continue until d_candidate_set not changes.
	//However, considering efficiency, we can terminate it in a few rounds
	int round = 5;
	//the last round is just a check, not refining
	for(int k = 0; k < round; ++k)
	{
		for(int i = 0; i < simple_num; ++i)
		{
			unsigned u = simple_vertices[i];
			unsigned* d_array = NULL;
			unsigned d_array_num = 0;
			kernel_collect(d_candidate_set+dsize*u, d_array, d_array_num);
			if(d_array_num == 0)
			{
				cout<<"no candidate for the "<<u<<"th query vertex"<<endl;
				return false; 
			}
			if(k == round - 1)
			{
				/*cout<<"check cand: "<<u<<" "<<d_array_num<<endl;*/
				checkCudaErrors(cudaFree(d_array));
				continue;
			}
			kernel_refine(i, d_array, d_array_num, d_candidate_set, simple_vertices, simple_edges);
			checkCudaErrors(cudaFree(d_array));
		}
	}

	return true;
}

__global__ void
count_kernel(unsigned* d_data_row_offset, unsigned* d_data_column_index, bool* d_candidate_set, int dsize, unsigned* d_array, unsigned d_array_num, int u, int v, unsigned* d_count)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int group = t / 32;
	int idx = t % 32;
	if(group >= d_array_num)
	{
		return; 
	}
	unsigned mu = d_array[group];
	unsigned begin = d_data_row_offset[mu];
	unsigned size = d_data_row_offset[mu+1] - begin;
	unsigned loop = size / 32;
	size = size % 32;

	unsigned count = 0;
	//each thread get adjacent vertex of mu as mv
	unsigned base = 0;
	for(int j = 0; j < loop; ++j, base+=32)
	{
		unsigned mv = d_data_column_index[begin+idx+base];
		if(d_candidate_set[dsize*v+mv])
		{
			count++;
		}
	}
	if(idx < size)
	{
		unsigned mv = d_data_column_index[begin+idx+base];
		if(d_candidate_set[dsize*v+mv])
		{
			count++;
		}
	}
	//reduce sum in a warp
	//REFERENCE: https://blog.csdn.net/bruce_0712/article/details/64926471
	for(int offset = 16; offset > 0; offset >>= 1)
	{
		count += __shfl_down(count, offset);
	}
	if(idx == 0)
	{
		d_count[group] = count;
	}
}

void 
Match::kernel_count(int u, int v, unsigned* d_array, unsigned d_array_num, bool* d_candidate_set, unsigned* d_count)
{
	int dsize = this->data->vertex_num;
	//assign a warp to each candidate vertex
	int WARP_SIZE = 32;
	int BLOCK_SIZE = 256;
	/*int BLOCK_SIZE = 64;*/
	int GRID_SIZE = (d_array_num*WARP_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
	/*cout<<"check array num: "<<u<<" "<<d_array_num<<" "<<GRID_SIZE<<" "<<WARP_SIZE<<endl;*/
	count_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_data_row_offset, d_data_column_index, d_candidate_set, dsize, d_array, d_array_num, u, v, d_count);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

__global__ void 
examine_kernel(unsigned* d_data_row_offset, unsigned* d_data_column_index, bool* d_candidate_set, int dsize, unsigned d_array_num, int u, int v, unsigned* d_tmp)
{
	//NOTICE: we must use volatile restriction to force each thread to read from shared memory instead of local register
	/*extern __volatile__ __shared__ unsigned write_pos[];*/
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int group = t / 32;
	int idx = t % 32;
	if(group >= d_array_num)
	{
		return; 
	}
	unsigned mu = d_tmp[group];
	//set shared memory content
	/*printf("group: %d\n", group);*/
	//just use __shfl instead of shared memory
	/*write_pos[group] = d_tmp[group+d_array_num];*/
	unsigned write_pos = d_tmp[group+d_array_num];
	unsigned begin = d_data_row_offset[mu];
	unsigned size = d_data_row_offset[mu+1] - begin;
	unsigned loop = size / 32;
	size = size % 32;

	//each thread get adjacent vertex of mu as mv
	unsigned base = 0;
	unsigned pred = 0, presum = 0;
	unsigned mv;
	for(int j = 0; j < loop; ++j, base+=32)
	{
		mv = d_data_column_index[begin+idx+base];
		pred = d_candidate_set[dsize*v+mv];
		presum = pred;
		//BETTER: extract into a device function
		//prefix sum in a warp
		for(unsigned stride = 1; stride <= idx; stride <<= 1)
		{
			presum += __shfl_up(presum, stride);
		}
		unsigned total = __shfl(presum, 31);  //broadcast to all threads in the warp
		//transform inclusive prefixSum to exclusive prefixSum
		presum = __shfl_up(presum, 1);
		//NOTICE: for the first element, the original presum value is copied
		if(idx == 0)
		{
			presum = 0;
		}
		//write to corresponding position
		//NOTICE: warp divergence exists(even we use compact, the divergence also exists in the compact operation)
		if(pred == 1)
		{
			d_tmp[2*d_array_num+1+write_pos+presum] = mv;
		}
		write_pos += total;
	}
	presum = pred = 0;
	if(idx < size)
	{
		mv = d_data_column_index[begin+idx+base];
		pred = d_candidate_set[dsize*v+mv];
		presum = pred;
	}
	//prefix sum in a warp
	for(unsigned stride = 1; stride <= idx; stride <<= 1)
	{
		presum += __shfl_up(presum, stride);
	}
	//transform inclusive prefixSum to exclusive prefixSum
	presum = __shfl_up(presum, 1);
	//NOTICE: for the first element, the original presum value is copied
	if(idx == 0)
	{
		presum = 0;
	}
	//write to corresponding position
	//NOTICE: warp divergence exists(even we use compact, the divergence also exists in the compact operation)
	if(pred == 1)
	{
		d_tmp[2*d_array_num+1+write_pos+presum] = mv;
	}
}

void 
Match::kernel_examine(int u, int v, unsigned d_array_num, bool* d_candidate_set, unsigned* d_tmp)
{
	int dsize = this->data->vertex_num;
	//assign a warp to each candidate vertex
	int WARP_SIZE = 32;
	int BLOCK_SIZE = 256;
	int GRID_SIZE = (d_array_num*WARP_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
	int SHARED_SIZE = sizeof(unsigned)*BLOCK_SIZE/WARP_SIZE;
	/*cout<<"shared size: "<<SHARED_SIZE<<endl;*/
	examine_kernel<<<GRID_SIZE, BLOCK_SIZE, SHARED_SIZE>>>(d_data_row_offset, d_data_column_index, d_candidate_set, dsize, d_array_num, u, v, d_tmp);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

bool
Match::FindCandidateEdges(bool* d_candidate_set, unsigned**& d_candidate_edge, unsigned*& d_candidate_edge_num)
{
	checkCudaErrors(cudaMalloc(&d_candidate_edge, sizeof(unsigned*)*this->edge_num));
	checkCudaErrors(cudaMalloc(&d_candidate_edge_num, sizeof(unsigned)*this->edge_num));
	unsigned* h_candidate_edge_num = new unsigned[this->edge_num];
	unsigned** h_candidate_edge = new unsigned*[this->edge_num];
	int dsize = this->data->vertex_num;

	for(int i = 0; i < this->edge_num; ++i)
	{
		unsigned u = this->edge_from[i];
		unsigned v = this->edge_to[i];
		unsigned* d_array = NULL;
		unsigned d_array_num = 0;
		kernel_collect(d_candidate_set+dsize*u, d_array, d_array_num);
		unsigned* d_count = NULL;
		checkCudaErrors(cudaMalloc(&d_count, sizeof(unsigned)*(d_array_num+1)));
		//two-step output scheme: find matching num
		//just like explore kernel, and do sum reduce in a warp
		kernel_count(u, v, d_array, d_array_num, d_candidate_set, d_count);
		
		//do prefix sum on d_count to find position
		thrust::device_ptr<unsigned> dev_ptr(d_count);
		unsigned sum;
		thrust::exclusive_scan(dev_ptr, dev_ptr+d_array_num+1, dev_ptr);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpy(&sum, d_count+d_array_num, sizeof(unsigned), cudaMemcpyDeviceToHost);
		unsigned* d_tmp = NULL;
		checkCudaErrors(cudaMalloc(&d_tmp, sizeof(unsigned)*(d_array_num*2+1+sum)));
		cudaMemcpy(d_tmp, d_array, sizeof(unsigned)*d_array_num, cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_tmp+d_array_num, d_count, sizeof(unsigned)*(d_array_num+1), cudaMemcpyDeviceToDevice);
		checkCudaErrors(cudaFree(d_array));
		checkCudaErrors(cudaFree(d_count));
		
		//two-step output scheme: re-examine the candidate edges and write them to the corresponding address of the hash table
		kernel_examine(u, v, d_array_num, d_candidate_set, d_tmp);

		h_candidate_edge_num[i] = d_array_num;
		h_candidate_edge[i] = d_tmp;

	}
	cudaMemcpy(d_candidate_edge_num, h_candidate_edge_num, sizeof(unsigned)*this->edge_num, cudaMemcpyHostToDevice);
	cudaMemcpy(d_candidate_edge, h_candidate_edge, sizeof(unsigned*)*this->edge_num, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	delete[] h_candidate_edge_num;
	delete[] h_candidate_edge;

	return true;
}

__global__ void
table_kernel(unsigned* d_candidate, unsigned* d_result, unsigned result_row_num, unsigned result_col_num, unsigned array_num)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int group = t / 32;
	int idx = t % 32;
	if(group >= array_num)
	{
		return; 
	}
	unsigned mu = d_candidate[group];
	unsigned begin = d_candidate[array_num+group];
	unsigned size = d_candidate[array_num+group+1] - begin;
	unsigned loop = size / 32;
	size = size % 32;
	//each thread get adjacent vertex of mu as mv
	unsigned base = begin;
	for(int j = 0; j < loop; ++j, base+=32)
	{
		unsigned mv = d_candidate[idx+base+2*array_num+1];
		//write mu and mv and to result table
		d_result[2*(base+idx)] = mu;
		d_result[2*(base+idx)+1] = mv;
	}
	if(idx < size)
	{
		unsigned mv = d_candidate[idx+base];
		d_result[2*(base+idx)] = mu;
		d_result[2*(base+idx)+1] = mv;
	}
}

__global__ void
join_kernel(unsigned* d_candidate, unsigned* d_result, unsigned* d_count, unsigned result_row_num, unsigned result_col_num, unsigned upos, unsigned vpos, unsigned array_num)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int group = t / 32;
	int idx = t % 32;
	if(group >= result_row_num)
	{
		return; 
	}
	
	unsigned mu = d_result[group*result_col_num+upos];
	unsigned mv = d_result[group*result_col_num+vpos];
	//find mu in d_array using a warp
	unsigned size = array_num;
	unsigned loop = size / 32;
	size = size % 32;

	unsigned valid_mu = 0;
	unsigned base = 0;
	for(int j = 0; j < loop; ++j, base+=32)
	{
		unsigned cand = d_candidate[idx+base];
		if(cand == mu)
		{
			valid_mu = 1;
		}
		if(__any(valid_mu) == 1)
		{
			break;
		}
	}
	if(valid_mu == 0 && idx < size)
	{
		unsigned cand = d_candidate[idx+base];
		if(cand == mu)
		{
			valid_mu = 1;
		}
	}
	//find the index of element mu
	unsigned idx_mu = __ballot(valid_mu);   //only one 1 in this situation
	//TODO: get the idx for each thread
	//n&(-n) to get the maxium num(which is 2's power), which can divide n (just like 10000..)
	//it is ok to add a log2() to get the idx
	valid_mu = __any(valid_mu);
	if(valid_mu == 0)
	{
		d_count[group] = 0;
		return; 
	}

	//find mv in adjs of mu using a warp
	unsigned valid_mv = 0;
	base = d_candidate[array_num+idx_mu];
	size = d_candidate[array_num+idx_mu+1] - base;
	loop = size / 32;
	size = size % 32;
	base = base + 2*array_num+1;
	for(int j = 0; j < loop; ++j, base+=32)
	{
		unsigned cand = d_candidate[idx+base];
		if(cand == mv)
		{
			valid_mv = 1;
		}
		if(__any(valid_mv) == 1)
		{
			break;
		}
	}
	if(valid_mv == 0 && idx < size)
	{
		unsigned cand = d_candidate[idx+base];
		if(cand == mv)
		{
			valid_mv = 1;
		}
	}
	valid_mv = __any(valid_mv);

	if(idx == 0)
	{
		d_count[group] = valid_mv;
	}
}

__global__ void
link_kernel(unsigned* d_candidate, unsigned* d_result, unsigned* d_result_new, unsigned* d_count, unsigned result_row_num, unsigned result_col_num, unsigned upos, unsigned vpos, unsigned array_num)
{

}

void 
Match::kernel_join(unsigned* d_candidate, unsigned*& d_result, unsigned& result_row_num, unsigned& result_col_num, unsigned upos, unsigned vpos, unsigned array_num)
{
	//follow the two-step output scheme to write the merged results
	unsigned* d_count = NULL;
	checkCudaErrors(cudaMalloc(&d_count, sizeof(unsigned)*(result_row_num+1)));
	join_kernel<<<(result_row_num*32+1023)/1024,1024>>>(d_candidate, d_result, d_count, result_row_num, result_col_num, upos, vpos, array_num);
	cudaDeviceSynchronize();
	
	//TODO: prefix sum to find position
	thrust::device_ptr<unsigned> dev_ptr(d_count);
	unsigned sum;
	thrust::exclusive_scan(dev_ptr, dev_ptr+result_row_num+1, dev_ptr);
	checkCudaErrors(cudaGetLastError());
	cudaMemcpy(&sum, d_count+result_row_num, sizeof(unsigned), cudaMemcpyDeviceToHost);
	if(sum == 0)
	{
		checkCudaErrors(cudaFree(d_count));
		checkCudaErrors(cudaFree(d_result));
		d_result = NULL;
		result_row_num = 0;
		return;
	}

	unsigned* d_result_new = NULL;
	checkCudaErrors(cudaMalloc(&d_result_new, sizeof(unsigned)*sum*result_col_num));
	link_kernel<<<(result_row_num*32+1023)/1024,1024>>>(d_candidate, d_result, d_result_new, d_count, result_row_num, result_col_num, upos, vpos, array_num);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaFree(d_count));

	checkCudaErrors(cudaFree(d_result));
	d_result = d_result_new;
	//NOTICE: result_col_num not changes in this case
	result_row_num = sum;
	//TODO: release resources
}

bool
Match::JoinCandidateEdges(unsigned** d_candidate_edge, unsigned* d_candidate_edge_num, unsigned*& d_result, unsigned& result_row_num, unsigned& result_col_num)
{
	int qsize = this->query->vertex_num;
	/*int dsize = this->data->vertex_num;*/
	bool* visited_vertices = new bool[qsize];
	bool* visited_edges = new bool[this->edge_num];
	memset(visited_vertices, false, sizeof(bool)*qsize);
	memset(visited_edges, false, sizeof(bool)*this->edge_num);

	//copy the candidate information to CPU memory
	unsigned* h_candidate_edge_size = new unsigned[this->edge_num];
	unsigned* h_candidate_edge_num = new unsigned[this->edge_num];
	cudaMemcpy(h_candidate_edge_num, d_candidate_edge_num, sizeof(unsigned)*this->edge_num, cudaMemcpyDeviceToHost);
	unsigned** h_candidate_edge = new unsigned*[this->edge_num];
	cudaMemcpy(h_candidate_edge, d_candidate_edge, sizeof(unsigned*)*this->edge_num, cudaMemcpyDeviceToHost);
	for(int i = 0; i < this->edge_num; ++i)
	{
		unsigned array_num = h_candidate_edge_num[i];
		unsigned* d_candidate = h_candidate_edge[i];
		cudaMemcpy(&h_candidate_edge_size[i], d_candidate+2*array_num, sizeof(unsigned), cudaMemcpyDeviceToHost);
	}

	//construct the result table: start from the smallest edge candidates
	unsigned minv = this->data->eSize()+1, minx = 0;
	for(unsigned i = 0; i < this->edge_num; ++i)
	{
		if(h_candidate_edge_size[i] < minv)
		{
			minv = h_candidate_edge_size[i];
			minx = i;
		}
	}
	result_col_num = 2;
	result_row_num = minv;
	visited_vertices[this->edge_from[minx]] = true;
	visited_vertices[this->edge_to[minx]] = true;
	visited_edges[minx] = true;
	checkCudaErrors(cudaMalloc(&d_result, sizeof(unsigned)*result_col_num*result_row_num));
	unsigned array_num = h_candidate_edge_num[minx];
	unsigned* d_candidate = h_candidate_edge[minx];
	//copy the candidates to build the intermediate table
	table_kernel<<<(array_num*32+1023)/1024,1024>>>(d_candidate, d_result, result_row_num, result_col_num, array_num);
	cudaDeviceSynchronize();
	this->add_mapping(this->edge_from[minx]);
	this->add_mapping(this->edge_to[minx]);

	for(int edge_cnt = 1; edge_cnt < this->edge_num; ++edge_cnt)
	{
		//choose next edge to join: two cases
		vector<unsigned> next_edges;
		bool expand_mode = false;   // join mode, two vertices connected, need to filter
		for(int i = 0; i < this->edge_num; ++i)
		{
			if(visited_edges[i])
			{
				continue;
			}
			if(visited_vertices[this->edge_from[i]] && visited_vertices[this->edge_to[i]])
			{
				next_edges.push_back(i);
			}
		}
		if(next_edges.empty())
		{
			expand_mode = true; // expand mode, only one vertex connected, just expand the result
			for(int i = 0; i < this->edge_num; ++i)
			{
				if(visited_edges[i])
				{
					continue;
				}
				next_edges.push_back(i);
			}
		}
		//if there are multiple such edges, select the one with the smallest number of candidates
		unsigned next_minv = this->data->eSize()+1, next_minx = 0;
		for(unsigned i = 0; i < next_edges.size(); ++i)
		{
			int edge_id = next_edges[i];
			if(h_candidate_edge_size[edge_id] < next_minv)
			{
				next_minv = h_candidate_edge_size[edge_id];
				next_minx = edge_id;
			}
		}
		d_candidate = h_candidate_edge[next_minx];
		//each warp deals with a row
		if(expand_mode)
		{
			//what should do if only v is connected?
			expand_kernel();
			//TODO: we need to add this mapping
			/*this->add_mapping(this->edge_to[minx]);*/
		}
		else
		{
			//both u and v are connected, then select u as the linking point and check v
			int u = this->edge_from[next_minx], v = this->edge_to[next_minx];
			array_num = h_candidate_edge_num[next_minx];
			kernel_join(d_candidate, d_result, result_row_num, result_col_num, this->id2pos[u], this->id2pos[v], array_num);
		}
		//TODO:cache the row in shared memory
	}

	delete[] h_candidate_edge;
	delete[] h_candidate_edge_num;
	delete[] h_candidate_edge_size;
	delete[] visited_vertices;
	delete[] visited_edges;
	return true;
}

void 
Match::match(IO& io, unsigned*& final_result, unsigned& result_row_num, unsigned& result_col_num, int*& id_map)
{
	long t0 = Util::get_cur_time();
	copyGraphToGPU();
	long t1 = Util::get_cur_time();
	cerr<<"copy graph used: "<<(t1-t0)<<"ms"<<endl;
#ifdef DEBUG
	cout<<"graph copied to GPU"<<endl;
#endif

	int qsize = this->query->vertex_num;
	int dsize = this->data->vertex_num;
	//NOTICE: an undirected edge is kept twice in the CSR format, but we only use one here
	this->edge_num = this->query->eSize() / 2;
	this->edge_from = new unsigned[edge_num];
	this->edge_to = new unsigned[edge_num];
	//check all in edges in the original directed query graph to avoid duplicates
	int edge_id = 0;
	vector<Vertex>& qin = this->query->vertices;
	for(int i = 0; i < qin.size(); ++i)
	{
		vector<Neighbor>& adj = qin[i].in;
		for(int j = 0; j < adj.size(); ++j)
		{
			//NOTICE: we treat it as undirected graph, and ensure from < to here
			this->edge_from[edge_id] = min(adj[j].vid, i);
			this->edge_to[edge_id] = max(adj[j].vid, i);
			edge_id++;
		}
	}

	//generate spanning Tree
	unsigned* filter_order = new unsigned[qsize];
	Graph* span_tree = new Graph;
	generateSpanningTree(span_tree, filter_order);

	bool* d_candidate_set = NULL;
	cudaMalloc(&d_candidate_set, sizeof(bool)*qsize*dsize);
	//filter out the candidate vertices
	bool success = InitializeCandidateVertices(span_tree, filter_order, d_candidate_set);
	if(!success)
	{
		final_result = NULL;
		result_row_num = 0;
		result_col_num = qsize;
		//TODO: release resources
		return; 
	}

	//refine the candidate vertices recursively
	RefineCandidateVertices(d_candidate_set);

	unsigned** d_candidate_edge = NULL;
	unsigned* d_candidate_edge_num = NULL;
	//gather candidates for edges
	FindCandidateEdges(d_candidate_set, d_candidate_edge, d_candidate_edge_num);

	//initialize the mapping structure
	this->id2pos = new int[qsize];
	this->pos2id = new int[qsize];
	this->current_pos = 0;
	memset(id2pos, -1, sizeof(int)*qsize);
	memset(pos2id, -1, sizeof(int)*qsize);

	unsigned* d_result = NULL;
	//join all candidate edges to get result
	JoinCandidateEdges(d_candidate_edge, d_candidate_edge_num, d_result, result_row_num, result_col_num);

	long t8 = Util::get_cur_time();
	//transfer the result to CPU and output
	if(success)
	{
		final_result = new unsigned[result_row_num * result_col_num];
		cudaMemcpy(final_result, d_result, sizeof(unsigned)*result_col_num*result_row_num, cudaMemcpyDeviceToHost);
	}
	else
	{
		final_result = NULL;
		result_row_num = 0;
		result_col_num = qsize;
	}
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	/*cudaFree(d_result);*/
	long t9 = Util::get_cur_time();
	cerr<<"copy result used: "<<(t9-t8)<<"ms"<<endl;
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	id_map = this->id2pos;

	//TODO: release resources
	delete span_tree;
	delete[] filter_order;
	cudaFree(d_candidate_set);
	release();
}

void
Match::release()
{
	/*delete[] this->id2pos;*/
	delete[] this->pos2id;
	//release query graph on GPU
	/*cudaFree(d_query_vertex_num);*/
	/*cudaFree(d_query_label_num);*/
	/*cudaFree(d_query_vertex_value);*/
	/*cudaFree(d_query_row_offset_in);*/
	/*cudaFree(d_query_row_offset_out); */
	/*cudaFree(d_query_edge_value_in);*/
	/*cudaFree(d_query_edge_offset_in);*/
	/*cudaFree(d_query_edge_value_out);*/
	/*cudaFree(d_query_edge_offset_out);*/
	/*cudaFree(d_query_column_index_in);*/
	/*cudaFree(d_query_column_index_out);*/
	/*cudaFree(d_query_inverse_label);*/
	/*cudaFree(d_query_inverse_offset);*/
	/*cudaFree(d_query_inverse_vertex);*/
	//release data graph on GPU
	/*cudaFree(d_data_vertex_num);*/
	/*cudaFree(d_data_label_num);*/
	/*cudaFree(d_data_vertex_value);*/
	cudaFree(d_data_row_offset);
	cudaFree(d_data_column_index);

#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
}

