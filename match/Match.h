/*=============================================================================
# Filename: Match.h
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 22:55
# Description: find all subgraph-graph mappings between query graph and data graph
=============================================================================*/

//HELP:
//nvcc compile: http://blog.csdn.net/wzk6_3_8/article/details/15501931
//cuda-memcheck:  http://docs.nvidia.com/cuda/cuda-memcheck/#about-cuda-memcheck
//nvprof:  http://blog.163.com/wujiaxing009@126/blog/static/71988399201701310151777?ignoreua
//
//Use 2D array on GPU:
//http://blog.csdn.net/lavorange/article/details/42125029
//
//http://blog.csdn.net/langb2014/article/details/51348523
//to see the memory frequency of device
//nvidia-smi -a -q -d CLOCK | fgrep -A 3 "Max Clocks" | fgrep "Memory"
//
//GPU cache:
//http://blog.csdn.net/langb2014/article/details/51348616

#ifndef _MATCH_MATCH_H
#define _MATCH_MATCH_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>

#include "gputimer.h"

#include "../util/Util.h"
#include "../graph/Graph.h"
#include "../io/IO.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

//Shared Memory and Memory Banks:  
//https://www.cnblogs.com/1024incn/p/4605502.html

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

//struct bool2uint_functor
//{
	////const float a;
	////bool2uint_functor(float _a): a(_a) {}

	//__host__ __device__ 
		//unsigned operator()(const bool& x, const unsigned& y) const
		//{
			//return x?1:0;
		//}
//};


class Match
{
public:
	Match(Graph* _query, Graph* _data);
	void match(IO& io, unsigned*& final_result, unsigned& result_row_num, unsigned& result_col_num, int*& id_map);
	~Match();

	static void initGPU(int dev);

private:
	Graph* query;
	Graph* data;

	int current_pos;
	int* id2pos;
	int* pos2id;
	void add_mapping(int _id);

	unsigned *d_query_vertex_num, *d_query_label_num, *d_query_vertex_value, *d_query_row_offset_in, *d_query_row_offset_out, *d_query_edge_value_in, *d_query_edge_offset_in, *d_query_edge_value_out, *d_query_edge_offset_out, *d_query_column_index_in, *d_query_column_index_out, *d_query_inverse_label, *d_query_inverse_offset, *d_query_inverse_vertex;
	unsigned *d_data_vertex_num, *d_data_label_num, *d_data_row_offset, *d_data_column_index, *d_data_vertex_value;

	void copyGraphToGPU();
	void release();

	//main routine of the GpSM algorithm
	void generateSpanningTree(Graph* span_tree, unsigned* filter_order);
	bool InitializeCandidateVertices(Graph*, unsigned*, bool* d_candidate_set);
	bool RefineCandidateVertices(bool* d_candidate_set);
	unsigned edge_num, *edge_from, *edge_to;  //edge info of query graph, from < to
	bool FindCandidateEdges(bool* d_candidate_set, unsigned**& d_candidate_edge, unsigned*& d_candidate_edge_num, bool v2u = false);
	bool JoinCandidateEdges(unsigned** d_candidate_edge, unsigned* d_candidate_edge_num, unsigned** d_candidate_edge_reverse, unsigned* d_candidate_edge_num_reverse, unsigned*& d_result, unsigned& result_row_num, unsigned& result_col_num);
	void generateSimplifiedGraph(std::vector<unsigned>& simple_vertices, std::vector< std::vector<unsigned> >& simple_edges, int degree_threshold);

	//kernel functions
	void kernel_check(int u, bool* valid);
	void kernel_collect(bool* valid, unsigned*& d_array, unsigned& d_array_num);
	void kernel_explore(int u, unsigned* d_array, unsigned d_array_num, bool* d_candidate_set, Graph* span_tree, bool* initialized);
	void kernel_refine(int u, unsigned* d_array, unsigned d_array_num, bool* d_candidate_set, std::vector<unsigned>& simple_vertices, std::vector< std::vector<unsigned> >& simple_edges);
	void kernel_count(int u, int v, unsigned* d_array, unsigned d_array_num, bool* d_candidate_set, unsigned* d_count);
	void kernel_examine(int u, int v, unsigned d_array_num, bool* d_candidate_set, unsigned* d_tmp);
	void kernel_join(unsigned* d_candidate, unsigned*& d_result, unsigned& result_row_num, unsigned& result_col_num, unsigned upos, unsigned vpos, unsigned array_num);
	void kernel_expand(unsigned* d_candidate, unsigned*& d_result, unsigned& result_row_num, unsigned& result_col_num, unsigned pos, unsigned array_num);

    //check functions
    bool checkEdge(unsigned** d_candidate_edge, unsigned* d_candidate_edge_num, int from, int to, int check_from, int check_to);
};

#endif

