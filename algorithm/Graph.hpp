//
//  Graph.hpp
//  algorithm
//
//  Created by shiwei on 2018/5/8.
//

//所有和图算法相关的题目

//1. 邻接点使用所用索引的好处是，所有遍历相关的参数可以使用索引来定位，而不用map,起点也使用索引。用索引定位就可以用数组来标记，而图的大小是固定的，所以数组大小固定，省去伸缩的消耗

/*
 经典问题：
 1. 连通分量/只求个数/是否联通
 2. 生成树
 3. 最短路径/最远的点
 4. 是否有环/包含特定点的环路
 5. 拓扑排序
 
 */

#ifndef Graph_hpp
#define Graph_hpp

#include <stdio.h>
#include <vector>
#include <stack>

using namespace std;

#pragma mark - 图节点加入了自定义的变量

namespace TFDataStruct{
    
    //有向图节点
    template <class T>
    struct DirectedGraphNode {
    public:
        T val;
        DirectedGraphNode(T val):val(val){};
        //adjacent nodes
        vector<DirectedGraphNode *> adjNodes;
        int visit = 0;
    };
    
    //有向图
    template <class T>
    class DirectedGraph{
        
    public:
        typedef DirectedGraphNode<T> NodeType;
    private:
        
        void clearVisits(){
            for (auto &node : allNodes){
                node.visit = 0;
            }
        }
        
        vector<DirectedGraphNode<T> *> starts;
        inline void checkStarts(){
            for (int i = 0; i<allNodes.size(); i++){
                if (allNodes[i].fromCount == 0) {
                    starts.push_back(&allNodes[i]);
                }
            }
        }
        
        typedef void(*NodeVisitHandler)(NodeType *node, bool start, void *context);
        typedef bool(*NodeVisitHandler2)(NodeType *pre,NodeType *cur, void *context);
        //以start节点为起点遍历所有可到达的节点，这些节点按遍历的顺序存入数组，然后返回
        //visitIdx里存放的是节点下一个要访问的邻接节点的索引，初始为0;或说正在访问的节点索引+1.
        //visit==0：未访问；visit 属于[1, adjNodes.size()]：正在访问； visit>adjNodes.size(),已经访问结束。也就是visit和“白、灰、黑”的状态标记合并了
        static void DFSNode(NodeType *start, NodeVisitHandler handler, void *context, bool *cyclic){
            
            stack<NodeType *> path;
            path.push(start);
            
            while (!path.empty()) {
                
                NodeType *&cur = path.top();
                
                //从邻接节点里找一个白色
                NodeType *next = nullptr;
                int nextIdx = cur->visit;
                
                while (nextIdx<cur->adjNodes.size()) {
                    next = cur->adjNodes[nextIdx];
                    if (next->visit == 0) { //未访问
                        break;
                    }
                    
                    nextIdx++;
                    //正在访问中
                    if (cyclic && next->visit<=next->adjNodes.size()) {
                        *cyclic = true;
                    }
                }
                cur->visit = nextIdx+1;
                
                if (nextIdx == cur->adjNodes.size()) { //没有未访问的节点了
                    if (handler) handler(cur, false, context); //访问结束回调
                    path.pop();
                    continue;
                }
                
                path.push(next);
                if (handler) handler(next, true, context);  //访问开始回调
            }
        }
        
#define BFSSetLen(node, len) (node->visit = ((len)<<1)|1)
#define BFSGetLen(node) ((node->visit)>>1)
#define BFSIsVisited(node) (node->visit&1)
        
        //1. 返回最后一个节点 2.visit包含两部分：最后一位的0/1表示是否访问过，其他位表示距离起点的路径长度，即visit = (len<<1)|visitFlag;
        static NodeType* BFSNode(NodeType *start, NodeVisitHandler2 handler, void *context){
            
            queue<NodeType *> visitQ;
            visitQ.push(start);
            BFSSetLen(start, 0);
            
            NodeType *front = nullptr;
            while (!visitQ.empty()) {
                front = visitQ.front();
                visitQ.pop();
                int len = BFSGetLen(front);
                
                for (auto &n : front->adjNodes){
                    if (!BFSIsVisited(n)) {
                        BFSSetLen(n, len+1);
                        visitQ.push(n);
                        
                        if (handler){
                            //返回true，结束整个搜索过程
                            if (handler(front, n, context)) {
                                return n;
                            }
                        };
                    }
                }
            }
            
            return front;
        }
    
    public:
        vector<NodeType> allNodes;
        
        //由邻接矩阵创建图
        static DirectedGraph<T> *createWithMat(vector<T> &values, vector<vector<int>> &matrix){
            DirectedGraph<T> *graph = new DirectedGraph<T>();
            
            int size = (int)values.size();
            for (auto &v : values) {
                graph->allNodes.push_back(NodeType(v));
            }
            
            for (int i = 0; i<size; i++) {
                NodeType &curNode = graph->allNodes[i];
                for (int j = 0; j<size; j++) {
                    if (matrix[i][j]) {
                        curNode.adjNodes.push_back(&graph->allNodes[j]);
                    }
                }
            }
            
            return graph;
        }
        
        //由邻接表创建图
        static DirectedGraph<T> *createWithEdges(vector<T> &values, vector<vector<int>> &edges){
            DirectedGraph<T> *graph = new DirectedGraph<T>();
            
            int size = (int)values.size();
            for (auto &v : values) {
                graph->allNodes.push_back(NodeType(v));
            }
            
            for (int i = 0; i<size; i++) {
                NodeType &curNode = graph->allNodes[i];
                auto &row = edges[i];
                for (int &idx : row) {
                    curNode.adjNodes.push_back(&graph->allNodes[idx]);
                }
            }
            
            return graph;
        }
        
        static inline void topSortDFSHandler(NodeType *node, bool start, void *context){
            if (!start) {
                stack<NodeType *> *nodes = (stack<NodeType *> *)context;
                nodes->push(node);
            }
        }
        
        //拓扑排序
        vector<NodeType *> topSort(){
            
            clearVisits();

            stack<NodeType *> nodes;
            bool cyclic = false;
            for (auto &n : allNodes){
                if (n.visit == 0) {
                    DFSNode(&n, topSortDFSHandler, &nodes, &cyclic);
                    if (cyclic) {  //有环，没有拓扑排序
                        return {};
                    }
                }
            }
            
            vector<NodeType *> result;
            while (!nodes.empty()) {
                result.push_back(nodes.top());
                nodes.pop();
            }
            
            return result;
        }
        
        //是否有环
        bool isCyclic(){
            
            clearVisits();
            bool cyclic = false;
            
            for (auto &n : allNodes){
                if (n.visit == 0) {
                    DFSNode(&n, nullptr, nullptr, &cyclic);
                    if (cyclic) {  //有环，没有拓扑排序
                        return true;
                    }
                }
            }
            
            return false;
        }
        
        static inline void isConnectedHandler(NodeType *node, void *context){
            int *count = (int*)context;
            *count = *count+1; //记录访问点的数量
        }

        
        //这个算法只适用于无向图
//        bool isConnected(){
//            clearVisits();
//
//            //任取一点，如果遍历完全部点，代表连通
//            int count = 0;
//            DFSNode(&allNodes.front(), isConnectedHandler, &count, nullptr);
//
//            return count==allNodes.size();
//        }
        

        
        static inline bool longestNodeHandler(NodeType *pre, NodeType *cur, void *context){
            int len = cur->visit>>1;
            printf("(%d len: %d) ",cur->val, len);
            return true;
        }
        //最远的点
        NodeType *longestNode(NodeType *start, int *length){
            clearVisits();
            auto last = BFSNode(start, longestNodeHandler, nullptr);
            if (length) {
                *length = BFSGetLen(last);
            }
            return last;
        }
        
        static inline bool shortestPathHandler(NodeType *pre, NodeType *cur, void *context){
            pair<unordered_map<NodeType*, NodeType *>&, NodeType*> *ctx = (pair<unordered_map<NodeType*, NodeType *>&, NodeType*>*)context;
            
            ctx->first[cur] = pre;
            
            if (cur == ctx->second) {
                return true;
            }
            
            return false;
        }
        
        //最短路径
        vector<NodeType *>shortestPath(NodeType *start, NodeType *end){
            clearVisits();
            
            unordered_map<NodeType*, NodeType *>froms;
            pair<unordered_map<NodeType*, NodeType *>&, NodeType*>context = {froms, end};
            
            auto last = BFSNode(start, shortestPathHandler, &context);
            if (last != end) { //从起点没找到终点
                return {};
            }
            
            int len = BFSGetLen(last);
            vector<NodeType*> path(len+1, nullptr);
            while (len>=0) {
                path[len] = last;
                last = froms[last];
                len--;
            }
            
            return path;
        }
        
        //遍历所有节点的一条路径,用拓扑排序
        vector<T> traversalPath(){
            
            if (allNodes.empty()) {
                return {};
            }
            
            checkStarts();
            //为空时，每个点都有前置，必定有环
            if (starts.empty()) {
                return {};
            }
            clearVisits();
            
            vector<DirectedGraphNode<T> *> path;
            vector<DirectedGraphNode<T> *> curSatrts = starts;
            //找出当前的起点，它一定是当前图最前面那个，然后把它放到最前面，再删掉它，形成新图再重复前面的操作。
            //实际操作时，只需要控制fromCount,维持一个fromCount为0的数组。
            while (!curSatrts.empty()) {
                auto cur = curSatrts.back();
                path.push_back(cur);
                curSatrts.pop_back();
                
                for (auto to : cur->adjNodes){
                    to->mark++;
                    if (to->mark == to->fromCount) {
                        curSatrts.push_back(to);
                    }
                }
            }
            
            //有环，没能遍历所有
            if (path.size() != allNodes.size()) {
                return {};
            }
            
            vector<T> result;
            for (auto node : path){
                result.push_back(node->val);
            }
            
            return result;
        }
        
    };
}

//bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
//    TFDataStruct::DirectedGraph<int> courses;
//    for (int i = 0; i<numCourses; i++) {
//        courses.allNodes.push_back(TFDataStruct::DirectedGraphNode<int>(i));
//    }
//
//    for (auto &pair : prerequisites){
//        courses.allNodes[pair.first].adjNodes.push_back(&courses.allNodes[pair.second]);
//        courses.allNodes[pair.second].fromCount++;
//    }
//
//    return courses.canTraversalAllAndNoCycle();
//}

//605. 序列重构
//1. 即使最后不按图的方式去做，也可以用图的方式来帮助理解
//2. 两个在某个子序列里相邻的点，在原序列里的关系只有: a) 直接相连 b)隔了多个点相连。
//3. 两个在原序列直接相连的点，在子序列里必定有一次是直接相连的，否则它们俩的关系是不确定的，那么就不唯一了。
//4. 结合2、3,在子序列里相邻的点，在原序列里，如果直接相连，则正确，它们的相对位置确定。如果隔了多个点，但方向对的，则待定。如果方向反了，则肯定错误。然后统计一定正确的数量，保证原序列的相连点都满足4.
bool sequenceReconstruction(vector<int> &org, vector<vector<int>> &seqs) {
    
    size_t maxNum = org.size();
    
    if (maxNum == 0) {
        for (int i = 0; i<seqs.size(); i++) {
            vector<int> &oneSeq = seqs[i];
            if (!oneSeq.empty()) {
                return false;
            }
        }
        return true;
    }else if (maxNum == 1) {
        bool exist = false;
        for (int i = 0; i<seqs.size(); i++) {
            vector<int> &oneSeq = seqs[i];
            if (oneSeq.size() > 1) {
                return false;
            }else if (oneSeq.size() == 1){
                if (oneSeq.front() > maxNum) {
                    return false;
                }else{
                    exist = true;
                }
            }
        }
        
        return exist;
    }
    
    //数字从1开始，多申请一个位置，避免后面频繁减1
    int checkedCount = 0;
    bool adjacent[maxNum+1];
    memset(adjacent, 0, sizeof(adjacent));
    
    int pos[maxNum+1];
    for (int i = 0; i<maxNum; i++) {
        pos[org[i]] = i;
    }
    
    for (int i = 0; i<seqs.size(); i++) {
        vector<int> &oneSeq = seqs[i];
        
        //先把第一个验了，后面只需要验证j+1，那么j也都被验证了
        if (!oneSeq.empty() && oneSeq.front() > maxNum) {
            return false;
        }
        for (int j = 0; j<(int)oneSeq.size()-1; j++) {
            if (oneSeq[j+1] > maxNum) {  //数超出去了，失败
                return false;
            }
            
            int curNum = oneSeq[j];
            int dis = pos[oneSeq[j+1]] - pos[curNum];
            if (dis<0) {   //顺序反了，失败
                return false;
            }else if (dis == 1 && !adjacent[curNum]){  //每新验证一个相邻的，计数并标记
                checkedCount++;
                adjacent[curNum] = true;
            }
        }
    }
    
    return checkedCount == maxNum-1;
}

//vector<int> findOrder(int numCourses, vector<pair<int, int>> &prerequisites) {
//    TFDataStruct::DirectedGraph<int> courses;
//    for (int i = 0; i<numCourses; i++) {
//        courses.allNodes.push_back(TFDataStruct::DirectedGraphNode<int>(i));
//    }
//
//    for (auto &pair : prerequisites){
//        courses.allNodes[pair.second].adjNodes.push_back(&courses.allNodes[pair.first]);
//        courses.allNodes[pair.first].fromCount++;
//    }
//    
//    return courses.traversalPath();
//}

#pragma mark - 使用最基本的图结构

namespace TFDataStruct_graph2{
    
    //有向图节点
    template <class T>
    struct DirectedGraphNode {
    public:
        T val;
        DirectedGraphNode(T val):val(val){};
        //adjacent nodes
        vector<DirectedGraphNode *> adjNodes;
    };
    
    //有向图
    template <class T>
    class DirectedGraph{
        
    public:
        typedef DirectedGraphNode<T> NodeType;
    private:
        
        typedef void(*NodeVisitHandler)(NodeType *node, bool start, void *context);
        //以start节点为起点遍历所有可到达的节点，这些节点按遍历的顺序存入数组，然后返回
        //visitIdx里存放的是节点下一个要访问的邻接节点的索引，初始为0;或说正在访问的节点索引+1.
        static void DFSNode(NodeType *start, unordered_map<NodeType *, int> &visitIdx, NodeVisitHandler handler, void *context, bool *cyclic){
            
            stack<NodeType *> path;
            path.push(start);
            
            while (!path.empty()) {
                
                NodeType *&cur = path.top();
                
                //从邻接节点里找一个白色
                NodeType *next = nullptr;
                int nextIdx = visitIdx[cur];
                
                while (nextIdx<cur->adjNodes.size()) {
                    next = cur->adjNodes[nextIdx];
                    if (visitIdx[next] == 0) { //未访问
                        break;
                    }
                    
                    nextIdx++;
                    //正在访问中
                    if (cyclic && visitIdx[next]<=next->adjNodes.size()) {
                        *cyclic = true;
                    }
                }
                visitIdx[cur] = nextIdx+1;
                
                if (nextIdx == cur->adjNodes.size()) { //没有未访问的节点了
                    if (handler) handler(cur, false, context); //访问结束回调
                    path.pop();
                    continue;
                }
                
                path.push(next);
                if (handler) handler(next, true, context);  //访问开始回调
            }
        }
        
    public:
        vector<NodeType> allNodes;
        
        //由邻接矩阵构建邻接表的图
        static DirectedGraph<T> *createAdj(vector<T> &values, vector<vector<int>> &matrix){
            DirectedGraph<T> *graph = new DirectedGraph<T>();
            
            int size = (int)values.size();
            for (auto &v : values) {
                graph->allNodes.push_back(NodeType(v));
            }
            
            for (int i = 0; i<size; i++) {
                NodeType &curNode = graph->allNodes[i];
                for (int j = 0; j<size; j++) {
                    if (matrix[i][j]) {
                        curNode.adjNodes.push_back(&graph->allNodes[j]);
                    }
                }
            }
            
            return graph;
        }
        
        static inline void topSortDFSHandler(NodeType *node, bool start, void *context){
            if (!start) {
                stack<NodeType *> *nodes = (stack<NodeType *> *)context;
                nodes->push(node);
            }
        }
        
        //拓扑排序
        vector<NodeType *> topSort(){
            
            unordered_map<NodeType *, int> visitState;
            
            stack<NodeType *> nodes;
            bool cyclic = false;
            for (auto &n : allNodes){
                if (!visitState[&n]) {
                    DFSNode(&n, visitState, topSortDFSHandler, &nodes, &cyclic);
                    if (cyclic) {  //有环，没有拓扑排序
                        return {};
                    }
                }
            }
            
            vector<NodeType *> result;
            while (!nodes.empty()) {
                result.push_back(nodes.top());
                nodes.pop();
            }
            
            return result;
        }
        
        //是否有环
        bool isCyclic(){
            unordered_map<NodeType *, int> visitState;
            
            stack<NodeType *> nodes;
            bool cyclic = false;
            
            for (auto &n : allNodes){
                if (!visitState[&n]) {
                    DFSNode(&n, visitState, nullptr, nullptr, &cyclic);
                    if (cyclic) {  //有环，没有拓扑排序
                        return true;
                    }
                }
            }
            
            return false;
        }
    };
}


#pragma mark - 无向图,邻接表/矩阵+索引类型,带权

namespace TFDataStruct{
    
    //已知起点，只记录权和另一点
    struct UndirectedGraphEdge1 {
        int other;
        int cost;
    };
    
    //包含两端和权
    struct UndirectedGraphEdge2 {
        int first;
        int second;
        int cost;
        
        bool operator<(const UndirectedGraphEdge2 &other) const{
            return this->cost<other.cost;
        }
    };
    
    //无向图节点
    template <class T>
    struct UndirectedGraphNode {
        T val;
        UndirectedGraphNode(T val):val(val){};
        //adjacent nodes
        vector<UndirectedGraphEdge1> edges;
    };
    
    static const int UndirectedGraphInfinity = INFINITY;
    
    template<class T>
    class UndirectedGraph {
    public:
        typedef UndirectedGraphNode<T> NodeType;
        
    private:
        //默认为空，调用genEdges，从邻接矩阵里获取所有边
        void genEdges(vector<UndirectedGraphEdge2> &allEdges){
            
            int size = (int)matrix.size();
            //维持i<j，避免重复计算
            for (int i = 0; i<size; i++) {
                for (int j = i+1; j<size; j++) {
                    if (matrix[i][j] != UndirectedGraphInfinity) {
                        allEdges.push_back({i, j, matrix[i][j]});
                    }
                }
            }
        }
        
    public:
        //所有顶点
        vector<NodeType> allNodes;
        
        //邻接矩阵(无穷大表示无边，其他代表权值)
        vector<vector<int>> matrix;
        void initMatrix(int size){
            for (int i = 0; i<size; i++) {
                matrix.push_back(vector<int>(size, 0));
                for (int j = 0; j<size; j++) {
                    matrix[i][j] = UndirectedGraphInfinity;
                }
            }
        }
        /*
         因为生成树包含所有顶点，所以顶点数据不需要记录，只需要记录边；然后除了根，每个顶点都对应一个边，所以可以使用数组记录,假设数组为A，则(A[i]--i)表示生成树的一条边。
         如果是用邻接矩阵表示的，则不需要输出边的权值，因为可以通过矩阵数组很快的(O(1))求得。
         */
        
        //prim算法求最小生成树
        vector<UndirectedGraphEdge1> lowestCost_prim(int root = 0){
            int size = (int)allNodes.size();

            //0:代表已处理，忽略；[1, int_max)，正在候选；int_max 没有访问到，暂不处理
            int distance[size];
            for (int i = 0; i<size; i++) {
                distance[i] = INT_MAX;
            }
            
            int processedCount = 1;
            vector<UndirectedGraphEdge1> closest(size,{0,0});
            
            NodeType *cur = &allNodes[root];
            int curIdx = root;
            distance[root] = 0;
            
            while (1) {
                //更新未处理点的距离
                for (auto &edge : cur->edges){
                    int dis = distance[edge.other];
                    if (edge.cost < dis) {
                        distance[edge.other] = edge.cost;
                        closest[edge.other] = {curIdx, edge.cost};
                    }
                }
                
                int next = -1, minCost = INT_MAX;
                for (int i = 0; i<size; i++) {
                    int dis = distance[i];
                    if (dis!=0 && dis!=INT_MAX) {
                        if (dis < minCost) {
                            minCost = dis;
                            next = i;
                        }
                    }
                }
                
                if (next<0) {
                    break;
                }
                
//                cout<<"in ("<<allNodes[closest[next].other].val<<", "<<allNodes[next].val<<") "<<endl;
                
                //找到下一个点
                cur = &allNodes[next];
                curIdx = next;
                distance[next] = 0;
                processedCount++;
            }
            
            if (processedCount < size) { //不连通，无法构造生成树
                return {};
            }
            return closest;
        }
        
        vector<UndirectedGraphEdge2> lowestCost_kruskal(int root = 0){
            
            vector<UndirectedGraphEdge2> allEdges;
            genEdges(allEdges);
            
            sort(allEdges.begin(), allEdges.end());
            
            int size = (int)allNodes.size();
            int connectIdx[size];
            for (int i = 0; i<size; i++) {
                connectIdx[i] = i;
            }
            
            vector<UndirectedGraphEdge2> result;
            
            for (auto &edge : allEdges) {
                
                if (connectIdx[edge.first] == connectIdx[edge.second]) {
                    continue;
                }
                
                for (int i = 0; i<size; i++) {
                    if (connectIdx[i] == connectIdx[edge.second]) {
                        connectIdx[i] = connectIdx[edge.first];
                    }
                }
                
                result.push_back(edge);
                if (result.size() == size-1) {
                    break;
                }
            }
            
            return result;
        }
    };
}


#endif /* Graph_hpp */
