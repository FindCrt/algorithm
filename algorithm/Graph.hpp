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
    class DirectedGraphNode {
    
        
    public:
        T val;
        DirectedGraphNode(T val):val(val){};
        vector<DirectedGraphNode *> toNodes;
        int fromCount = 0;
        
        short mark = 0;  //每次遍历的临时性标记
        size_t checking = 0;  //深度搜索时，遍历到第几个邻接节点了
        
        friend ostream& operator<<(ostream &os, DirectedGraphNode<T> &node){
            return os<<node.val;
        }
    };
    
    //有向图
    template <class T>
    class DirectedGraph{
        
        //有环导致错误
        const static int DGErrorCycle = -1;
        
        void clearMarks(){
            for (auto &node : allNodes){
                node.mark = 0;
                node.checking = 0;
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
        
        //以start节点为起点遍历所有可到达的节点，插入到path前面，如果正确返回0，错误返回对应错误码
        int DPSNodes(DirectedGraphNode<T> *start, vector<DirectedGraphNode<T> *> &path){
            vector<DirectedGraphNode<T> *> thisPath;
            
            stack<DirectedGraphNode<T> *> forks;
            DirectedGraphNode<T> *cur = start;
            cur->mark = 1;
            
            while (1) {
                //从邻接节点里找一个白色
                while (cur->checking < cur->toNodes.size()){
                    auto to = cur->toNodes[cur->checking];
                    cur->checking++;
                    if (to->mark == 0) { //没访问的
                        forks.push(cur);
                        
                        cur = to;
                        break;
                    }else if (to->mark == 1){  //正在进行的，现在又转回去了，说明有环
                        return DGErrorCycle;
                    }
                }
                
                //没找到白色子节点，返回
                if (cur->mark != 0) {
                    
                    while (!forks.empty() && cur->checking >= cur->toNodes.size()) {
                        cur->mark = 2;
                        cur = forks.top();
                        forks.pop();
                        
                    }
                    if (cur->checking >= cur->toNodes.size()) {
                        break;
                    }
                }else{
                    thisPath.push_back(cur);
                    cur->mark = 1;
                }
            }
            
            cout<<"start: "<<*start<<endl;
            for (auto &node : thisPath){
                cout<<*node<<" ";
            }
            cout<<endl;
            
            path.insert(path.begin(), thisPath.begin(), thisPath.end());
            return 0;
        }
        
    public:
        vector<DirectedGraphNode<T>> allNodes;
        
        //能否遍历所有且无环;可有多个起点，可以并行
        bool canTraversalAllAndNoCycle(){
            
            if (allNodes.empty()) {
                return true;
            }
            
            checkStarts();
            //为空时，每个点都有前置，必定有环
            if (starts.empty()) {
                return false;
            }
            clearMarks();
            
            int findCount = 0;
            for (auto start : starts){
                
                stack<DirectedGraphNode<T> *> path;
                DirectedGraphNode<T> *cur = start;
                
                findCount++;
                cur->mark = 1;
                
                //标记：0白色: 没访问; 1灰色: 开始访问还没结束; 2黑色: 访问了它之后所有节点并反悔了，结束。
                while (1) {
                    
                    //从邻接节点里找一个白色
                    while (cur->checking < cur->toNodes.size()){
                        auto to = cur->toNodes[cur->checking];
                        cur->checking++;
                        if (to->mark == 0) { //没访问的
                            path.push(cur);
                            
                            cur = to;
                            break;
                        }else if (to->mark == 1){  //正在进行的，现在又转回去了，说明有环
                            return false;
                        }
                    }
                    
                    //没找到白色子节点，返回
                    if (cur->mark != 0) {
                        
                        while (!path.empty() && cur->checking >= cur->toNodes.size()) {
                            cur->mark = 2;
                            cur = path.top();
                            path.pop();
                            
                        }
                        if (cur->checking >= cur->toNodes.size()) {
                            break;
                        }
                    }else{
                        
                        findCount++;
                        cur->mark = 1;
                    }
                }
            }
            
            return findCount == allNodes.size();
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
            clearMarks();
            
            vector<DirectedGraphNode<T> *> path;
            vector<DirectedGraphNode<T> *> curSatrts = starts;
            //找出当前的起点，它一定是当前图最前面那个，然后把它放到最前面，再删掉它，形成新图再重复前面的操作。
            //实际操作时，只需要控制fromCount,维持一个fromCount为0的数组。
            while (!curSatrts.empty()) {
                auto cur = curSatrts.back();
                path.push_back(cur);
                curSatrts.pop_back();
                
                for (auto to : cur->toNodes){
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

bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
    TFDataStruct::DirectedGraph<int> courses;
    for (int i = 0; i<numCourses; i++) {
        courses.allNodes.push_back(TFDataStruct::DirectedGraphNode<int>(i));
    }
    
    for (auto &pair : prerequisites){
        courses.allNodes[pair.first].toNodes.push_back(&courses.allNodes[pair.second]);
        courses.allNodes[pair.second].fromCount++;
    }
    
    return courses.canTraversalAllAndNoCycle();
}

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

vector<int> findOrder(int numCourses, vector<pair<int, int>> &prerequisites) {
    TFDataStruct::DirectedGraph<int> courses;
    for (int i = 0; i<numCourses; i++) {
        courses.allNodes.push_back(TFDataStruct::DirectedGraphNode<int>(i));
    }
    
    for (auto &pair : prerequisites){
        courses.allNodes[pair.second].toNodes.push_back(&courses.allNodes[pair.first]);
        courses.allNodes[pair.first].fromCount++;
    }
    
    return courses.traversalPath();
}

#endif /* Graph_hpp */
