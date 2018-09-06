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
#include "LRUCache.hpp"

#include "TopK.hpp"
#include "Graph.hpp"
#include "MultiwayTree.hpp"
#include "BinaryTree.hpp"
#include "LFUCache.hpp"
#include "SegmentTree.hpp"
#include "CommonStructs.hpp"

#define LFUCache(n) auto cache = LFUCache(n);
#define set(a,b) printf("set(%d) ",a);cache.set(a,b);cout<<cache<<endl;
#define get(a) printf("get(%d) %d\n",a,cache.get(a));cout<<cache<<endl;

//248. 统计比给定整数小的数的个数
//使用线段树解觉，关键是线段树的节点的值的定义：是在这个范围内的数的个数。
//解就转化为了在[0, queries[i]-1]这个范围内的数的个数
vector<int> countOfSmallerNumber(vector<int> &A, vector<int> &queries) {
    typedef TFDataStruct::SegmentTree<int, sumMergeFunc,short> MyTree;
    
    vector<short> count(10001, 0);
    for (auto &num : A){
        count[num]++;
    }
    
    auto root = MyTree::build(count);
    vector<int> result;
    for (auto &query : queries){
        result.push_back(MyTree::query(root, 0, query-1));
    }
    
    return result;
}

//249. 统计前面比自己小的数的个数
vector<int> countOfSmallerNumberII(vector<int> &A) {
    typedef TFDataStruct::SegmentTree<int, sumMergeFunc,short> MyTree;
    
    auto root = MyTree::build(0, 10001);
    
    vector<int> result;
    for (auto &num : A){
        MyTree::add(root, num, num, 1);
        result.push_back(MyTree::query(root, 0, num-1));
    }
    
    return result;
}

//360. 滑动窗口的中位数
vector<int> medianSlidingWindow(vector<int> &nums, int k) {
    if (nums.empty()) {
        return {};
    }
    if (k == 1) {
        return nums;
    }
    
    map<int, int> leftNums;
    map<int, int> rightNums;
    int median = nums.front();
    leftNums[median] = 1;
    
    int leftSize = 1, rightSize = 0;
    
    vector<int> result;
    
    for (int i = 1; i<nums.size(); i++) {
        int j =  i-k+1;
        
        if (j>0) {
            int drop = nums[j-1];
            if (drop>median) {
                if (--rightNums[drop] == 0) {
                    rightNums.erase(drop);
                }
                rightSize--;
            }else{
                if (--leftNums[drop] == 0) {
                    leftNums.erase(drop);
                }
                leftSize--;
            }
        }
        
        int insert = nums[i];
        if (insert>median) {
            rightNums[insert]++;
            rightSize++;
        }else{
            leftNums[insert]++;
            leftSize++;
        }
        
        while (1) {
            int minPartSize = (leftSize+rightSize-1)/2; //中位数之前的数量，这个一个基准线
            
            int medianCount = leftNums[median];
            if (leftSize-medianCount > minPartSize) { //小于中位数的已经超过了前面的数量
                leftNums.erase(median);
                rightNums[median] = medianCount;
                median = leftNums.rbegin()->first;
                
                leftSize -= medianCount;
                rightSize += medianCount;
            }else if(leftSize<=minPartSize){  //小于和等于的数量之和没超过基准线，中位数要选更大的
                median = rightNums.begin()->first;
                int count = rightNums.begin()->second;
                rightNums.erase(rightNums.begin());
                
                leftNums[median] = count;
                
                leftSize += count;
                rightSize -= count;
            }else{
                break;
            }
        }
        
        if (j>=0) {
            result.push_back(median);
        }
    }
    
    return result;
}

vector<int> winSum(vector<int> &nums, int k) {
    if (nums.empty() || k == 0 || k > nums.size()) {
        return {};
    }
    if (k == 1) {
        return nums;
    }
    vector<int> result;
    
    int sum = 0;
    for (int i = 0; i<k; i++) {
        sum += nums[i];
    }
    result.push_back(sum);
    
    for (int i = k; i<nums.size(); i++) {
        sum += (nums[i]-nums[i-k]);
        result.push_back(sum);
    }
    
    return result;
}

//362. 滑动窗口的最大值
//解法是维持一个队列，这个队列里的元素保持严格的单调递减，并且都是当前窗口内的元素。
//遇见一个新的元素，从队尾开始比较，把所有小于或等于它的元素都踢掉，然后存入。
//如果窗口移动导致丢弃的元素是最大值，就把第一个元素移除。
//巧妙的地方在于队列里的元素，左右关系跟原数组一样，又保持单调递减，所以最大值就是第一个，而且丢弃的元素也只可能是第一个。
//第二层循环的次数决定了时间复杂度，虽然不知怎么证明，但是统计一下随机数据，可以得到平均的次数为0.9多，数值并不重要，关键是它是稳定的，不会随着数据量的变大而变化。
vector<int> maxSlidingWindow(vector<int> nums, int k) {
    if (nums.empty() || k == 0 || k > nums.size()) {
        return {};
    }
    if (k == 1) {
        return nums;
    }
    
    vector<int> result;
    vector<pair<int, int>> maxNums;
    
    int popCount = 0;
    for (int i = 0; i<nums.size(); i++){
        
        int j = i-k+1;
        if (j>0 && maxNums.front().second == j-1) {
            maxNums.erase(maxNums.begin());
        }
        
        while (!maxNums.empty() && maxNums.back().first <= nums[i]) {
            maxNums.pop_back();
            popCount++;
        }
        
        maxNums.push_back({nums[i], i});
        
        if (j>=0) {
            result.push_back(maxNums.front().first);
        }
    }
    
    cout<<"popCount: "<<popCount/(float)nums.size()<<endl;
    
    return result;
}

int maxSlidingMatrix(vector<vector<int>> &matrix, int k) {
    if (k>matrix.size()) {
        return 0;
    }
}

#define LRUCache(c) LRUCache cache(c);
int main(int argc, const char * argv[]) {

}
