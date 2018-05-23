/*=============================================================================
# Filename: Graph.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 23:01
# Description: 
=============================================================================*/

#include "Graph.h"

using namespace std;

void 
Graph::addVertex(LABEL _vlb)
{
	this->vertices.push_back(Vertex(_vlb));
}

void 
Graph::addEdge(VID _from, VID _to, LABEL _elb)
{
	this->vertices[_from].out.push_back(Neighbor(_to, _elb));
	this->vertices[_to].in.push_back(Neighbor(_from, _elb));
}

//BETTER+TODO: construct all indices using GPU, with thrust or CUB, back40computing or moderngpu
//NOTICE: for data graph, this transformation only needs to be done once, and can match 
//many query graphs later
void 
Graph::transformToCSR()
{
	this->vertex_num = this->vertices.size();
	this->vertex_value = new unsigned[this->vertex_num];
	this->undirected_row_offset = new unsigned[this->vertex_num+1];
	this->undirected_row_offset[0] = 0;
	
	//NOTICE: we assume there is no circle like A->B and B->A (this case will bring in parallel edge in undirected graphs)
	int edge_num = 0, i, j;
	for(i = 0; i < this->vertex_num; ++i)
	{
		this->vertex_value[i] = this->vertices[i].label;
		int insize = this->vertices[i].in.size(), outsize = this->vertices[i].out.size();
		edge_num += insize;
		edge_num += outsize;
		this->undirected_row_offset[i+1] = this->undirected_row_offset[i] + insize + outsize;
	}
	this->undirected_column_index = new unsigned[edge_num];

	int pos = 0;
	for(i = 0; i < this->vertex_num; ++i)
	{
		int insize = this->vertices[i].in.size(), outsize = this->vertices[i].out.size();
		int begin = pos;
		for(j = 0; j < insize; ++j, ++pos)
		{
			Neighbor& tn = this->vertices[i].in[j];
			this->undirected_column_index[pos] = tn.vid;
		}
		for(j = 0; j < outsize; ++j, ++pos)
		{
			Neighbor& tn = this->vertices[i].out[j];
			this->undirected_column_index[pos] = tn.vid;
		}
		sort(this->undirected_column_index+begin, this->undirected_column_index+pos, less<unsigned>());
	}

	//to construct inverse label list
	Element* elelist = new Element[this->vertex_num];
	for(i = 0; i <this->vertex_num; ++i)
	{
		elelist[i].id = i;
		elelist[i].label = this->vertex_value[i];
	}
	sort(elelist, elelist+this->vertex_num);

	int label_num = 0;
	for(i = 0; i <this->vertex_num; ++i)
	{
		if(i == 0 || elelist[i].label != elelist[i-1].label)
		{
			label_num++;
		}
	}

	this->label_num = label_num;
	this->inverse_label = new unsigned[label_num];
	this->inverse_num = new unsigned[label_num+1];
	j = 0;
	int base = 0;
	for(i = 1; i < this->vertex_num; ++i)
	{
		if(elelist[i].label != elelist[i-1].label)
		{
			this->inverse_label[j] = elelist[i-1].label;
			this->inverse_num[j] = i - base;
			base = i;
			++j;
		}
	}
	this->inverse_label[j] = elelist[i-1].label;
	this->inverse_num[j] = i - base;

	delete[] elelist;
}

bool 
Graph::isEdgeContained(VID from, VID to, LABEL label)
{
	vector<Neighbor>& out = this->vertices[from].out;
	for(int i = 0; i < out.size(); ++i)
	{
		if(out[i].vid == to && out[i].elb == label)
		{
			return true;
		}
	}
	return false;
}

void 
Graph::printGraph()
{
	int i, n = this->vertex_num;
	cout<<"vertex value:"<<endl;
	for(i = 0; i < n; ++i)
	{
		cout<<this->vertex_value[i]<<" ";
	}cout<<endl;

	//check CSR 
	cout<<"row offset:"<<endl;
	for(i = 0; i <= n; ++i)
	{
		cout<<this->undirected_row_offset[i]<<" ";
	}cout<<endl;
	cout<<"column index in:"<<endl;
	for(i = 0; i < this->eSize(); ++i)
	{
		cout<<this->undirected_column_index[i]<<" ";
	}cout<<endl;

}

