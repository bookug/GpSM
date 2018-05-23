# GpSM

implementation of GpSM program by Ha-Nguyen to solve subgraph isomorphism problem on GPU

---

#### Dataset

注意：所有数据集的节点编号都必须从0开始，所有标签必须从1开始，
否则会有问题

we target at a one-to-one mapping at a time, the query graph is small(vertices less than 100), while the data graph can be very large.
(but can be placed in GPU's global memory)

目前的子图同构采用的是普通的子图的形式，而不是导出子图的形式(induced subgraph)

论文中默认处理的是无向的不带边标签的图，对于有向图或带边标签的图，均需要做修改(比如论文中用到的生成树)
或者先视为无向图用算法完成匹配，再对匹配的每条结果用真实完整的限制条件进行验证

默认图中不存在这种情况: A->B and B->A, 否则在视为无向图处理时，将出现平行边

---

#### Paper 

Fast Subgraph Matching on Large Graphs using Graphics Processors, DASFAA 2015 (CCF Database B class)

---

#### Algorithm

filter and verify framework

edge as the basic unit

joins candidate edges in parallel to form partial solutions during the verification phase

To solve the problem of  the considerable amount of intermediate results for
joining operations,  we adopt the pruning technique of "Relational consistency algorithms and their application in finding
subgraph and graph isomorphisms", 
further enhancing it by ignoring low-connectivity vertices 
which have little or no effect of decreasing intermediate results during filtering.

