/*=============================================================================
# Filename: Graph.h
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 23:00
# Description: 
=============================================================================*/

//Data Structure:  CSR with 4 arrays, we should use structure of array instead of array of structure
//row offsets
//column indices
//values(label of edge)
//flags(label of vertice)

#ifndef _GRAPH_GRAPH_H
#define _GRAPH_GRAPH_H

#include "../util/Util.h"

class Neighbor
{
public:
	VID vid;
	LABEL elb;
	Neighbor()
	{
		vid = -1;
		elb = -1;
	}
	Neighbor(int _vid, int _elb)
	{
		vid = _vid;
		elb = _elb;
	}
	//bool operator<(const Neighbor& _nb) const
	//{
		//return this->vid < _nb.vid;
	//}
	bool operator<(const Neighbor& _nb) const
	{
		if(this->elb == _nb.elb)
		{
			return this->vid < _nb.vid;
		}
		else
		{
			return this->elb < _nb.elb;
		}
	}
};

class Element
{
public:
	int label;
	int id;
	bool operator<(const Element& _ele) const
	{
		if(this->label == _ele.label)
		{
			return this->id <_ele.id;
		}
		else
		{
			return this->label < _ele.label;
		}
	}
};

class Vertex
{
public:
	//VID id;
	LABEL label;
	//NOTICE:VID and EID is just used in this single graph
	std::vector<Neighbor> in;
	std::vector<Neighbor> out;
	Vertex()
	{
		label = -1;
	}
	Vertex(LABEL lb):label(lb)
	{
	}
	inline unsigned eSize() const
	{
		return in.size()+out.size();
	}
};

class Graph
{
public:
	std::vector<Vertex> vertices;
	void addVertex(LABEL _vlb);
	void addEdge(VID _from, VID _to, LABEL _elb);

	//CSR format: 4 pointers
	unsigned vertex_num;
	unsigned* vertex_value;

	//structures for undirected graph
	unsigned* undirected_row_offset;
	unsigned* undirected_column_index;

	//unsigned* row_offset_in;  //range is 0~vertex_num, the final is a border(not valid vertex)
	//unsigned* edge_value_in;
	//unsigned* edge_offset_in;
	//unsigned* column_index_in;

	//unsigned* row_offset_out;
	//unsigned* edge_value_out;
	//unsigned* edge_offset_out;
	//unsigned* column_index_out;

	//Inverse Label List
	unsigned label_num;
	unsigned* inverse_label;
	unsigned* inverse_num;

	Graph() 
	{
		undirected_row_offset = undirected_column_index = NULL;
		vertex_num = 0;
		label_num = 0;
		inverse_label = inverse_num = NULL;
	}
	~Graph() 
	{ 
		delete[] undirected_row_offset;
		delete[] undirected_column_index;
		delete[] inverse_label;
		delete[] inverse_num;
	}

	void transformToCSR();
	bool isEdgeContained(VID from, VID to, LABEL label);
	void printGraph();

	inline unsigned eSize() const
	{
		return this->undirected_row_offset[vertex_num];
	}
	inline unsigned vSize() const
	{
		return vertex_num;
	}
	
	inline void getNeighbor(unsigned id, unsigned& neighbor_num, unsigned& neighbor_offset)
	{
		neighbor_offset = this->undirected_row_offset[id];
		neighbor_num = this->undirected_row_offset[id+1] - neighbor_offset;
	}
	inline int getVertexDegree(unsigned id)
	{
		return this->undirected_row_offset[id+1] - this->undirected_row_offset[id];
	}
};

#endif

