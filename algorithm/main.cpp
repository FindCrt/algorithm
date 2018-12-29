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
#include <queue>
#include <list>
#include <unordered_map>
#include <iostream>
#include "heap.hpp"
#include <mach/mach_time.h>
#include <unordered_set>
#include <fstream>
#include<stdlib.h>
//#include "page1.hpp"

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

void readArrays(){
    vector<int> nums;
    vector<string> names;
    vector<int> w;
    int state = 0;
    string path = "/Users/apple/Downloads/1000.in";
    
    readFile(path, [&state, &nums, &names, &w](string &line){
        
        //        auto cStr = line.c_str();
        int start = -1;
        for (int x = 0; x<line.size(); x++){
            //            float rate = (float)x/line.size();
            //            if (x%1000 == 0) {
            //                printf("rate: %d,%.3f\n",x,rate);
            //            }
            char c = line[x];
            if (c == '['){
                state++;
                start = x+1;
            }else if (c==','){
                
                if (state == 1) {
                    int num = 0, weight = 1;
                    for (int k = x-1; k>=start; k--) {
                        num += weight*(line[k]-'0');
                        weight *= 10;
                    }
                    nums.push_back(num);
                }else if (state == 2){
                    string name;
                    for (int k = start+1; k<x-1; k++) {
                        name.push_back(line[k]);
                    }
                    names.push_back(name);
                }else{
                    int num = 0, weight = 1;
                    for (int k = x-1; k>=start; k--) {
                        num += weight*(line[k]-'0');
                        weight *= 10;
                    }
                    w.push_back(num);
                }
                
                start = x+1;
            }
        }
    });
}

vector<ListNode*> binaryTreeToLists(TreeNode* root) {
    if (root == nullptr) {
        return {};
    }
    vector<ListNode *> result;
    vector<TreeNode *> last = {root};
    while (!last.empty()) {
        vector<TreeNode *> next;
        ListNode *rowHead = new ListNode(0), *cur = rowHead;
        
        for (auto &n : last){
            
            cur->next = new ListNode(n->val);
            cur = cur->next;
            
            if (n->left) {
                next.push_back(n->left);
            }
            if (n->right) {
                next.push_back(n->right);
            }
        }
        
        last = next;
        result.push_back(rowHead->next);
    }
    
    return result;
}

vector<vector<int>> levelOrder(TreeNode * root) {
    
    vector<vector<int>> result;
    if (root == nullptr) {
        return result;
    }
    
    queue<TreeNode *> iterNodes;
    iterNodes.push(root);
    iterNodes.push(nullptr);
    
    vector<int> planeVal;
    
    while (iterNodes.size() > 1) {
        
        auto &node = iterNodes.front();
        iterNodes.pop();
        if (node == nullptr) {
            
            result.push_back(planeVal);
            planeVal = {};
            iterNodes.push(nullptr);
            continue;
        }
        
        planeVal.push_back(node->val);
        
        if (node->left) {
            iterNodes.push(node->left);
        }
        if (node->right) {
            iterNodes.push(node->right);
        }
    }
    
    result.push_back(planeVal);
    return result;
}

//797. 到达一个数字
int reachNumber(int target) {
    // Write your code here
    target = abs(target);
    int k = 0;
    while (target > 0)
        target -= ++k;
    return target % 2 == 0 ? k : k + 1 + (k&1);
}

bool reachEndpoint(vector<vector<int>> &map) {
    vector<Point> path;
    bool able = canReachPoint(map, {0,0}, 0, 1, 9, &path);
    
    for (auto &p : path){
        cout<<"("<<p.x<<", "<<p.y<<") ";
    }
    return able;
}

//股票交易5，每天都可以进行最多一次交易，求最后的最大收益
//这里的思想是：使用动态规划思想，左侧[0,k-1]为以求最优解区域，现在考虑k位置。在前面k个里面找到一个“可以买入”且价格最低的时刻买入，然后今天卖出。把这次交易叠加到前面的最优解里就是当前的最优解。
//“可以买入”是指原本是卖出或不操作的时刻，卖出改为不操作或者不操作改为买入，效果都等价于买入。
//如果找不到这么一天，就不操作了。
int stockExchange5(vector<int> &a) {
    int result = 0;
    int process[a.size()];
    memset(process, 0, sizeof(process));
    
    for (int i = 1; i<a.size(); i++) {
        int minVal = a[i], minIdx = -1;
        for (int j = 0; j<i; j++) {
            if (a[j] < minVal && process[j] > -1) {
                minVal = a[j];
                minIdx = j;
            }
        }
        
        if (minIdx >= 0) {
            process[minIdx]--;
            process[i] = 1;
            
            result += a[i]-minVal;
        }
        
//        printf("result %d \n",result);
    }
    
    return result;
}

void combinationSum2(vector<int> &num, int target, int start, vector<int> &res, int sum, vector<vector<int>> &result){
    
    int last = -1;
    for (int i = start; i<num.size(); i++) {
        if (num[i] == last) {
            continue;
        }
        last = num[i];
        
        res.push_back(num[i]);
        if (sum == target-num[i]) {
            result.push_back(res);
        }
        combinationSum2(num, target, i+1, res, sum+num[i], result);
        res.pop_back();
    }
}

/*
 
 由递归转为非递归的操作步骤：
 1. 定义节点：把递归函数里的参数中变化的量加入，加入分支选择的变量；一般用到递归都是多分支的递归，下一级走到哪个分支需要记录
 2. 每个节点就对应着每次递归函数调用时的状态，也就是每个节点对应一个问题。
 3. 每个节点拿出来，主要任务就是做分支选择，也就是选择一个解，然后问题缩化为更小的问题。这题里就是拿到新的node.visit
 4. 栈回溯有两种情况： 1. 没有进一步的选择了 2. 得到解了
 
 总结起来：对于每个节点的处理流程是：
 1. 分支选择，位置1
 2. 没有选择出栈回溯，位置2
 3. 选择后当前数据的更新，同时也是求解的过程，位置3
 4. 由第3步分化：得到解，存储解，出栈回溯；没得到，把下一步问题压入栈，进入下一轮循环。
 
 主要核心就两个： 分支选择和求解，只是分支选择的意外情况是没得选了，回溯；求解的意外情况是得到解了，回溯。
 */
struct combinationSumNode{
    int start;
    int visit;
    int sum;
};

vector<vector<int>> combinationSum2(vector<int> &num, int target) {
    sort(num.begin(), num.end());
    printVectorOneLine(num);
    vector<int> res;
    vector<vector<int>> result;
//    combinationSum2(num, target, 0, res, 0, result);
    
    int size = (int)num.size();
    
    stack<combinationSumNode> path;
    path.push({0, -1, 0});
    
    while (!path.empty()) {
        //位置1
        auto &node = path.top();
        if (node.visit == -1) {
            node.visit = node.start;
        }else{
            node.sum -= num[node.visit];
            res.pop_back();
            do {
                node.visit++;
            } while (node.visit < size && num[node.visit] == num[node.visit-1]);
        }
        
        //位置2
        if (node.visit == size) {
            path.pop();
            continue;
        }
        
        int choice = num[node.visit];
        
        //位置3
        res.push_back(choice);
        node.sum += choice;
        printVectorOneLine(res);
        if (node.sum >= target) {
            if (node.sum == target) {
                result.push_back(res);
            }
            
            path.pop();
            res.pop_back();
        }else{
            //位置4
            path.push({ node.visit+1, -1, node.sum});
        }
    }
    
    return result;
}

#define LRUCache(c) LRUCache cache(c);
int main(int argc, const char * argv[]) {
    
//    uint64_t start = mach_absolute_time();
    
    vector<int> nums = {3,1,3,5,1,1};
    auto result = combinationSum2(nums, 8);
//    printf("%d \n",result);
    printTwoDVector(result);
    
//    uint64_t duration = mach_absolute_time() - start;
//    mach_timebase_info_data_t timebase;
//    mach_timebase_info(&timebase);
//    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
//    printf("exe time: %.1f ms\n",time);
//    printf("%d \n",result);
//    printVectorOneLine(result);
    
//    canGameOver(nums, 500549646);
}
