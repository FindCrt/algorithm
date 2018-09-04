//
//  main.m
//  algorithm
//
//  Created by shiwei on 17/8/16.
//
//

#include <stdio.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <stack>
#include <list>
#include <unordered_map>
#include <iostream>
#include "heap.hpp"
#include <mach/mach_time.h>
#include <unordered_set>
#include <fstream>

#include "TFSort.h"
#include "MinStack.hpp"

#include "TopK.hpp"
#include "Graph.hpp"
#include "MultiwayTree.hpp"
#include "BinaryTree.hpp"
#include "LFUCache.hpp"
#include "SegmentTree.hpp"
#include "CommonStructs.hpp"

#define LFUCache(n) auto cache = LFUCache(n);
#define set(a,b) cache.set(a,b);
#define get(a) printf("get(%d) %d\n",a,cache.get(a));

vector<int> resolveSegmentInput(string source, int size){
    vector<int> counts(size, 0);
    int mark = 0, firstNum = -1;
    bool isLeave = false;
    for (int i = 0; i<source.size(); i++) {
        char c = source[i];
        if (c == '[') {
            mark = i+1;
        }else if (c == ','){
            int num = atoi(source.substr(mark,i-mark).c_str());
            if (firstNum < 0) {
                firstNum = num;
                mark = i+1;
            }else if (firstNum == num){
                isLeave = true;
            }
        }else if (c == '='){
            mark = i+1;
        }else if (c == ']'){
            int num = atoi(source.substr(mark,i-mark).c_str());
            if (isLeave) {
                counts[firstNum] = num;
            }
            firstNum = -1;
        }
    }
    
    return counts;
}

//833. 进程序列
vector<int> numberOfProcesses(vector<Interval> &logs, vector<int> &queries) {
    // Write your code here
    vector<pair<long, long>> vec;
    for(auto e: logs)
    {
        vec.push_back(make_pair(e.start, 0));
        vec.push_back(make_pair(e.end+1, 1));
    }
    
    for(int i=0; i<queries.size(); i++)
    {
        vec.push_back(make_pair(queries[i], 2));
    }
    
    int count=0;
    vector<int> res(queries.size());
    sort(vec.begin(), vec.end());
    map<int, int> mp;
    
    
    for(int i=0; i<vec.size(); i++)
    {
        if(vec[i].second==0)
        {
            count++;
        }
        else if (vec[i].second==1)
        {
            count--;
        }
        else
        {
            mp[vec[i].first] = count;
        }
    }
    int i=0;
    for(auto e: queries)
    {
        res[i++] = mp[e];
    }
    return res;
}

//751. 约翰的生意 这题符合线段树的应用环境
//借一位刷题朋友的话：线段树的应用环境是构建时使用点粒度，查询时使用区间粒度。
//这题不仅应用环境很匹配，而且数据量也很大，别说暴力解不行，就是递归实现的线段树都不行
vector<int> business(vector<int> &A, int k) {
    typedef SegmentTree<int, minMergeFunc> MyTree;
    
    auto root = MyTree::build(A);
    vector<int> result;
    
    int size = (int)A.size();
    for (int i = 0; i<size; i++) {
        int minPrice = MyTree::query(root, max(i-k, 0), min(size, i+k), INT_MIN);
        result.push_back(max(0, A[i]-minPrice));
    }
    
    return result;
}

//205. 区间最小数
vector<int> intervalMinNumber(vector<int> &A, vector<Interval> &queries) {
    typedef SegmentTree<int, minMergeFunc> MyTree;
    
    auto root = MyTree::build(A);
    vector<int> result;
    
    for (auto &inter : queries) {
        int minNum = MyTree::query(root, inter.start, inter.end, INT_MIN);
        result.push_back(minNum);
    }
    
    return result;
}

//206. 区间求和 I
vector<long long> intervalSum(vector<int> &A, vector<Interval> &queries) {
    typedef SegmentTree<long long, sumMergeFunc, int> MyTree;
    
    MyTree::NodeType *root = MyTree::build(A);
    vector<long long> result;
    
    for (auto &inter : queries) {
        long long sum = MyTree::query(root, inter.start, inter.end, 0);
        result.push_back(sum);
    }
    
    return result;
}

int main(int argc, const char * argv[]) {
    vector<int> nums = {1,2,7,8,5};
    vector<Interval> queries;
    queries.push_back(Interval(1,2));
    auto result = intervalSum(nums, queries);
    printVectorOneLine(result);
}
