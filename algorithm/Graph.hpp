//
//  Graph.hpp
//  algorithm
//
//  Created by shiwei on 2018/5/8.
//

//所有和图算法相关的题目

#ifndef Graph_hpp
#define Graph_hpp

#include <stdio.h>
#include <vector>
#include <stack>

using namespace std;

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
    
    //无向图节点
    template <class T>
    struct UndirectedGraphNode {
        int val = 0;
        int mark = 0;
        vector<UndirectedGraphNode *> neighbors;
        
        UndirectedGraphNode *from = nullptr;
        
        UndirectedGraphNode(int val):val(val){};
    };
    
    //有向图
    template <class T>
    class DirectedGraph{
        
    public:
        typedef DirectedGraphNode<T> NodeType;
    private:
        //有环导致错误
        const static int DGErrorCycle = -1;
        
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
        
        bool isCyclic2(){
            
            clearVisits();
            
            stack<NodeType *> nodes;
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


#endif /* Graph_hpp */
