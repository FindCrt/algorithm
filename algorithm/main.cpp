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

//89. K数之和
//动态规划，思路：1. 通过一个什么样的操作，从大的问题转化为小的问题 2. 尽量转化为单问题，不要分散成多个子问题
int kSumIII(vector<int> &A, int k, int target) {
    
    if (A.size() < k) {
        return 0;
    }
    
    int result[k+1][target+1];
    for (int i = 0; i<k+1; i++) {
        memset(result[i], 0, sizeof(result[i]));
    }
    if (A.back()<=target) result[1][A.back()] = 1;
    
    for (int i = (int)A.size()-2; i>=0; i--) {
        int curNum = A[i];
        for (int x = k-1; x>=0; x--) {
            for (int y = target; y>=0; y--) {
                if (y+curNum<=target) {
                    //原来写的是++，当做增加一次了，其实会增加多次，原方案有N种，那么新增一个数后，对应方案应该也是增加N种。这里用的是乘法逻辑，两段路选择， 每段都有多种方案，就是两者方案数相乘。
                    //找出问题还是靠的跟随流程推到，即跟着一步步的走流程，这样把过程放慢了才找得到错误点。
                    result[x+1][y+curNum]+=result[x][y];
                }
            }
        }
        
        if (curNum<=target) result[1][curNum]++;
        
    }
    
    return result[k][target];
}

//126. 最大树
TreeNode * maxTree1(vector<int> &A) {
    if (A.empty()) {
        return nullptr;
    }
    TreeNode *pre = new TreeNode(A.front());
    if (A.size() == 1) {
        return pre;
    }
    
    map<TreeNode *, TreeNode *> parents;
    TreeNode *root = pre;
    
    for (int i = 1; i<A.size(); i++){
        int num = A[i];
        TreeNode *cur = new TreeNode(num);
        
        TreeNode *parent = pre;
        TreeNode *leftChild = nullptr;
        while (parent && parent->val < num) {
            leftChild = parent;
            parent = parents[parent];
        }
        
        if (parent) {
            parent->right = cur;
            parents[cur] = parent;
        }
        cur->left = leftChild;
        parents[leftChild] = cur;
        
        if (root == leftChild) {
            root = cur;
        }
        
        pre = cur;
    }
    
    return root;
}

TreeNode * maxTree(vector<int> &A) {
    if (A.empty()) {
        return nullptr;
    }
    
    stack<TreeNode *> rightBranchs;
    TreeNode *cur = new TreeNode(A.front());
    rightBranchs.push(cur);
    
    TreeNode *root = cur;
    
    for (int i = 1; i<A.size(); i++) {
        TreeNode *cur = new TreeNode(A[i]);
        
        TreeNode *top = nullptr;
        while (!rightBranchs.empty() && cur->val > rightBranchs.top()->val) {
            top = rightBranchs.top();
            rightBranchs.pop();
        }
        
        cur->left = top;
        if (rightBranchs.empty()) {
            root = cur;
        }else{
            rightBranchs.top()->right = cur;
        }
        rightBranchs.push(cur);
    }
    
    return root;
}

bool isMatch(string &s, string &p) {
    bool matchMap[s.size()+1][p.size()+1];
    for (int i = 0; i<=s.size(); i++) {
        memset(matchMap[i], 0, sizeof(matchMap[i]));
    }
    matchMap[s.size()][p.size()] = true;
    
    for (int i = (int)s.size()-1; i>=0; i--) {
        auto &sChar = s[i];
        bool isStar = false;
        for (int j = (int)p.size()-1; j>=0; j--) {
            
            if (isStar) {
                if (p[j] == '.') {
                    for (int k = i; k<=s.size(); k++) {
                        if (matchMap[k][j+2]) {
                            matchMap[i][j] = true;
                            printf(".*: p%d - s[%d,%d]\n",j,i,k-i);
                            break;
                        }
                    }
                }else{
                    int k = i;
                    do {
                        if (matchMap[k][j+2]) {
                            matchMap[i][j] = true;
                            printf("%c*: p%d - s[%d,%d]\n",p[j],j,i,k-i);
                            break;
                        }
                    } while (s[k++]==p[j]);
                }
                isStar = false;
            }else{
                if (p[j] == '*') {
                    isStar = true;
                    continue;
                }else if (p[j] == '.'){
                    matchMap[i][j] = matchMap[i+1][j+1];
                }else{
                    matchMap[i][j] = (p[j]==sChar) && matchMap[i+1][j+1];
                }
            }
        }
    }
    
    return matchMap[0][0];
}

//168. 吹气球
int maxCoins(vector<int> &nums) {
    
    int size = (int)nums.size();
    nums.insert(nums.begin(), 1);
    nums.push_back(1);
    
    int coins[size+2][size+2];
    for (int i = 0; i<size+2; i++) {
        memset(coins[i], 0, sizeof(coins[i]));
    }
    
    for (int len = 1; len<=size; len++) {
        for (int i = 1; i<=size-len+1; i++) {
            int j = i+len-1;
            for (int x = i; x<=j; x++) {
                coins[i][j] = max(coins[i][j], coins[i][x-1]+coins[x+1][j]+nums[i-1]*nums[x]*nums[j+1]);
            }
        }
    }
    
    return coins[1][size];
}

string binaryRepresentationInt(string intNum){
    int num = atoi(intNum.c_str());
    if (num == 0) {
        return "0";
    }
    
    string str = "";
    while (num > 0) {
        str.insert(str.begin(), (num&1)?'1':'0');
        cout<<str<<endl;
        num >>= 1;
    }
    
    return str;
}

string binaryRepresentationDecimal(string decimal){
    double decNum = atof(decimal.c_str());
    if (decNum == 0) {
        return "";
    }
    
    string str = ".";
    int count = 0;
    while (decNum != 0) {
        char next = (char)(decNum*2);
        str.push_back('0'+next);
        decNum = (decNum*2)-next;
        count++;
        if (count == 32) {
            return "ERROR";
        }
    }
    
    return str;
}

string binaryRepresentation(string &n) {
    auto dot = n.find('.');
    string decimalPart = binaryRepresentationDecimal(n.substr(dot, n.size()-dot));
    if (decimalPart.front() == 'E') {
        return "ERROR";
    }else{
        string intPart = binaryRepresentationInt(n.substr(0, dot));
        return intPart+decimalPart;
    }
}

//183. 木材加工
//使用二分法优化，关键在于单木头的长度和总个数是单调递减的关系，即木头约短，总个数越多。只要存在单调的关系，那么就有二分法的用武之地。
//这一题非常好的提醒了这一点
int woodCut(vector<int> &L, int k) {
    int maxLen = 0;
    int residue = k;
    for (int len : L){
        if (residue>0) {
            residue -=len;  //不直接比较总量是因为可能会超出int范围
        }
        maxLen = max(maxLen, len);
    }
    
    //排除头<k
    if (residue > 0) {
        return 0;
    }
    
    //排除尾>=k
    int maxLenCount = 0;
    for (int len : L){
        if (len == maxLen) {
            if ((++maxLenCount)==k) {
                return maxLen;
            }
        }
    }
    
    //循环不变条件是：左边结果>=k,右边<k。所以前面先把不满足的头和尾情况排除,逼近到最后left和right相邻时，left就是解。
    int left = 1, right = maxLen;
    while (left < right-1) {
        int mid = left+(right-left)/2;
        
        int count = 0;
        for (int len : L){
            count += len/mid;
        }
        if (count<k) {
            right = mid;
        }else{
            left = mid;
        }
    }
    
    return left;
}

#define LFUCache(n) auto cache = LFUCache(n);
#define set(a,b) cache.set(a,b);
#define get(a) printf("get(%d) %d\n",a,cache.get(a));

int main(int argc, const char * argv[]) {
    
    vector<int> nums = {2147483644,2147483645,2147483646,2147483647};
    
    auto result = woodCut(nums, 4);
    
    cout<<result<<endl;
}
