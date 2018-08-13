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

#include "TFSort.h"
#include "MinStack.hpp"

#include "page1.hpp"
#include "page2.hpp"
#include "TopK.hpp"
#include "Graph.hpp"
#include "MultiwayTree.hpp"
#include "BinaryTree.hpp"
#include "LFUCache.hpp"

using namespace std;

vector<pair<int, double>> dicesSum(int n) {

    vector<vector<double>> probs(n + 1, vector<double>(6 * n + 1));
    
    for (int i = 1; i<=6; i++) {
        probs[1][i] = 1.0/6.0;
    }
    
    for (int i = 2; i<=n; i++) {
        for (int j = i; j<=6*i; j++) {
            for (int k = 1; k<=6; k++) {
                if (j>k) {
                    probs[i][j] += probs[i-1][j-k];
                }
            }
            probs[i][j] /= 6.0;
        }
    }
    
    vector<pair<int, double>> result;
    for (int i = n; i<=6*n; i++) {
        result.push_back(make_pair(i, probs[n][i]));
    }
    
    return result;
}

vector<pair<int, double>> dicesSum2(int n) {
    // Write your code here
    vector<pair<int, double>> results;
    vector<vector<double>> f(n + 1, vector<double>(6 * n + 1));
    
    for (int i = 1; i <= 6; ++i) f[1][i] = 1.0 / 6;
    
    for (int i = 2; i <= n; ++i)
        for (int j = i; j <= 6 * i; ++j) {
            for (int k = 1; k <= 6; ++k)
                if (j > k)
                    f[i][j] += f[i - 1][j - k];
            f[i][j] /= 6.0;
        }
    
    for (int i = n; i <= 6 * n; ++i)
        results.push_back(make_pair(i, f[n][i]));
    
    return results;
}



//43. 最大子数组 III
//对于怎么建立DP的模型有参考价值,
//1. 不可重复，可以从解的形式出发，解有多个元素的时候，可以固定第一个元素，从而降级
//2. 要唯一性的降级，不要一个问题拆分之后变成多个问题，最好是一个问题变成一个更小的问题，一维的问题就从开头切割。
int maxSubArray(vector<int> &nums, int k) {
    int results[nums.size()][k+1];
    
    for (int i = 0; i<nums.size(); i++) {
        for (int j = 1; j<=k; j++) {
            results[i][j] = INT_MIN;
        }
    }
    
    results[nums.size()-1][1] = nums.back();
    
    cout<<"\n***********last"<<endl;
    for (int m = 0; m<k+1; m++) {
        cout<<results[nums.size()-1][m]<<" ";
    }
    
    vector<int> involves = {nums.back()};
    int involveMax = nums.back();
    
    for (int i = (int)nums.size()-2; i>=0; --i) {
        auto &resi = results[i];
        int curNum = nums[i];
        
        //involves是包含当前数的各个长度的区间和
        involves.insert(involves.begin(), 0);
        for (int r = 0; r < involves.size(); ++r){
            involves[r] += curNum;
        }
        
        involveMax = involveMax>0 ? curNum+involveMax : curNum;
        resi[1] = max(involveMax, results[i+1][1]);
        
        for (int j = 2; j<=min(k, (int)nums.size()-i); j++) {
            //第一个区间包含当前的时
            resi[j] = results[i+1][j];
            for (int r = 0; r < involves.size(); ++r){
                if (j<=nums.size()-i-r) {//剩余的数的数量>=区间数量
                    resi[j] = max(resi[j], involves[r]+results[i+r+1][j-1]);
                }
            }
        }
        
        cout<<"\n***********"<<i<<endl;
        for (int m = 0; m<k+1; m++) {
            cout<<resi[m]<<" ";
        }
    }
    
    return results[0][k];
}

#define INT_MAX_STR "2147483647"
#define INT_MIN_STR "-2147483648"

int atoi(string &str) {
    if (str.empty()) {
        return 0;
    }
    
    //去除头空格
    int start = 0;
    while (str[start] == ' ') {
        start++;
    }
    
    bool postive = str[start] != '-';
    if (!postive || str[start] == '+') {
        start += 1;
    }
    
    //检测非法
    int end = start;
    while (end<str.length() && str[end] <= '9' && str[end] >= '0') {
        end++;
    }
    
    int length = end-start;
    if (length > 10 || (length == 10 && str.substr(start,length) > INT_MAX_STR)) {
        return postive?INT_MAX:INT_MIN;
    }
    
    int result = 0;
    int bit = 1;
    for (int i = end-1;i>=start;i--){
        result += bit*(str[i]-'0');
        bit*=10;
    }
    
    if (!postive) {
        result *= -1;
    }
    
    return result;
}

//86. 二叉查找树迭代器
//O(h)的空间复杂度实现思路是，用一个h长度的栈来存储从root到当前节点的路径
//O(1)空间复杂度的思路是，把left节点指向parent,因为是中序遍历，使用深度搜索，搜索到某个节点时，左节点已经不需要了
class BSTIterator {
    TreeNode *cur = new TreeNode(0);
    TreeNode *parent = nullptr;
    
    inline void findTreeFirst(){
        auto left = cur->left;
        cur->left = parent;
        
        while (left) {
            parent = cur;
            cur = left;
            left = cur->left;
            cur->left = parent;
        }
    }
    
public:
    
    BSTIterator(TreeNode * root) {
        cur->right = root;
    }

    bool hasNext() {
        if (cur->right) {
            return true;
        }
        auto check = cur;
        while (check->left && check->left->right == check) {
            check = check->left; //实际是去到parent
        }
        return check->left != nullptr;
    }

    TreeNode * next() {
        if (cur->right) {
            parent = cur;
            cur = cur->right;
            findTreeFirst();
        }else{
            //cur是右孩子，就要一直向上追溯
            while (cur->left && cur->left->right == cur) {
                cur = cur->left; //实际是去到parent
            }
            cur = cur->left;
        }
        
        return cur;
    }
};

#define LFUCache(n) auto cache = LFUCache(n);
#define set(a,b) cache.set(a,b);
#define get(a) printf("get(%d) %d\n",a,cache.get(a));

int main(int argc, const char * argv[]) {

    vector<string> nodeChars = {"10","1","11","#","6","#","12"};
    TreeNode *root = TreeNode::createWithArray(nodeChars);
    
    BSTIterator iter = BSTIterator(root);
    while (iter.hasNext()) {
        printf("%d ",iter.next()->val);
    }
    printf("\n");
}
