/*=============================================================================
# Filename: IO.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 22:55
# Description: 
=============================================================================*/

#include "IO.h"

using namespace std;

IO::IO()
{
	this->qfp = NULL;
	this->dfp = NULL;
	this->ofp = NULL;
	this->data_id = -1;
}

IO::IO(string query, string data, string file)
{
	this->data_id = -1;
	this->line = "============================================================";
	qfp = fopen(query.c_str(), "r");
	if(qfp == NULL)
	{
		cerr<<"input open error!"<<endl;
		return;
	}
	dfp = fopen(data.c_str(), "r");
	if(dfp == NULL)
	{
		cerr<<"input open error!"<<endl;
		return;
	}
	ofp = fopen(file.c_str(), "w+");
	if(ofp == NULL)
	{
		cerr<<"output open error!"<<endl;
		return;
	}
}

Graph* 
IO::input(FILE* fp)
{
	char c1, c2;
	int id0, id1, id2, lb;
	bool flag = false;
	Graph* ng = NULL;

	while(true)
	{
		fscanf(fp, "%c", &c1);
		if(c1 == 't')
		{
			if(flag)
			{
				fseek(fp, -1, SEEK_CUR);
				return ng;
			}
			flag = true;
			fscanf(fp, " %c %d\n", &c2, &id0);
			if(id0 == -1)
			{
				return NULL;
			}
			else
			{
				ng = new Graph;
			}
			//read vertex num, edge num, vertex label num, edge label num
			int numVertex, numEdge, vertexLabelNum, edgeLabelNum;
			fscanf(fp, " %d %d %d %d\n", &numVertex, &numEdge, &vertexLabelNum, &edgeLabelNum);
		}
		else if(c1 == 'v')
		{
			fscanf(fp, " %d %d\n", &id1, &lb);
//NOTICE: we add 1 to labels for both vertex and edge, to ensure the label is positive!
			//ng->addVertex(lb+1); 
			ng->addVertex(lb); 
		}
		else if(c1 == 'e')
		{
			fscanf(fp, " %d %d %d\n", &id1, &id2, &lb);
			//NOTICE:we treat this graph as directed, each edge represents two
			//This may cause too many matchings, if to reduce, only add the first one
			//ng->addEdge(id1, id2, lb+1);
			ng->addEdge(id1, id2, lb);
			//ng->addEdge(id2, id1, lb);
		}
		else 
		{
			cerr<<"ERROR in input() -- invalid char"<<endl;
			return false;
		}
	}
	return NULL;
}

bool 
IO::input(Graph*& data_graph)
{
	data_graph = this->input(this->dfp);
	if(data_graph == NULL)
		return false;
	this->data_id++;
	data_graph->transformToCSR();
	return true;
}

bool 
IO::input(vector<Graph*>& query_list)
{
	Graph* graph = NULL;
	while(true)
	{
		graph = this->input(qfp);
		if(graph == NULL) //to the end
			break;
		graph->transformToCSR();
		query_list.push_back(graph);
	}

	return true;
}

bool 
IO::output(int qid)
{
	fprintf(ofp, "query graph:%d    data graph:%d\n", qid, this->data_id);
	fprintf(ofp, "%s\n", line.c_str());
	return true;
}

bool
IO::output()
{
	fprintf(ofp, "\n\n\n");
	return true;
}

//BETTER: move the process of verifying to the GPU(full CSR of the data graph is required)
bool
IO::verify(unsigned* ans, unsigned num, Graph* query, Graph* data)
{
	//check isomorphism
	set<unsigned> uniq_set;
	for(int i = 0; i < num; ++i)
	{
		if(uniq_set.find(ans[i]) != uniq_set.end())
		{
#ifdef DEBUG
			//cout<<"not unique "<<ans[i]<<endl;
#endif
			return false;
		}
		uniq_set.insert(ans[i]);
	}
	//enumerate each edge in the query graph to check if this is a valid mapping
	vector<Vertex>& qvlist = query->vertices;
	for(int i = 0; i < qvlist.size(); ++i)
	{
		//NOTICE: we only need to check all incoming edges to avoid duplicates
		vector<Neighbor>& in = qvlist[i].in;
		for(int j = 0; j < in.size(); ++j)
		{
			int to = ans[i];
			int from = ans[in[j].vid];
			int label = in[j].elb;
			bool flag = data->isEdgeContained(from, to, label);
			if(!flag)
			{
				return false;
			}
		}
	}
	return true;
}

bool
IO::output(unsigned* final_result, unsigned result_row_num, unsigned result_col_num, int* id_map, Graph* query_graph, Graph* data_graph)
{
	cout<<"result: "<<result_row_num<<" "<<result_col_num<<endl;
	int i, j, k;
	for(i = 0; i < result_row_num; ++i)
	{
		unsigned* ans = final_result + i * result_col_num;
#ifdef DEBUG
        //cout<<ans[id_map[0]]<<" "<<ans[id_map[1]]<<" "<<ans[id_map[2]]<<endl;
#endif
		bool valid = verify(ans, result_col_num, query_graph, data_graph);
		if(!valid)
		{
#ifdef DEBUG
			//cout<<"a result is invalid!"<<endl;
#endif
			continue;
		}

		for(j = 0; j < result_col_num; ++j)
		{
			k = ans[id_map[j]];
			fprintf(ofp, "(%u, %u) ", j, k);
		}
		fprintf(ofp, "\n");
	}
	fprintf(ofp, "\n\n\n");
	return true;
}

bool 
IO::output(int* m, int size)
{
	for(int i = 0; i < size; ++i)
	{
		fprintf(ofp, "(%d, %d) ", i, m[i]);
	}
	fprintf(ofp, "\n");
	return true;
}

void
IO::flush()
{
	fflush(this->ofp);
}

IO::~IO()
{
	fclose(this->qfp);
	this->qfp = NULL;
	fclose(this->dfp);
	this->dfp = NULL;
	fclose(this->ofp);
	this->ofp = NULL;
}

