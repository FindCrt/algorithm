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
#include "TypicalProblems.hpp"
#include "DeapFirstSearch.hpp"
//#include "page1.hpp"

#include "TFSort.h"
#include "MinStack.hpp"
#include "LRUCache.hpp"

//#include "TopK.hpp"
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
 准确的说，这是广义的深度搜索的步骤，不是递归的。
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

void solveNQueens(int n, int idx, vector<vector<string>> &result, vector<int> &placed){
    if (idx == n) {
        vector<string> ans(n, string(n, '.'));
        for (int i = 0; i<n; i++) {
            ans[i][placed[i]] = 'Q';
        }
        result.push_back(ans);
        return;
    }
    
    bool forbidden[n];
    memset(forbidden, 0, sizeof(forbidden));
    
    for (int i = 0; i<idx; i++) {
        forbidden[placed[i]] = true;
        int left = placed[i]-(idx-i), right = placed[i]+(idx-i);
        if (left >= 0) {
            forbidden[left] = true;
        }
        if (right < n) {
            forbidden[right] = true;
        }
    }
    
    for (int i = 0; i<n; i++) {
        if (!forbidden[i]) {
            placed[idx] = i;
            solveNQueens(n, idx+1, result, placed);
        }
    }
}

vector<vector<string>> solveNQueens(int n){
    vector<int> placed(n, 0);
    vector<vector<string>> result;
    solveNQueens(n, 0, result, placed);
    
    return result;
}


struct NQueenNode {
    int idx;
    bool *forbidden;
};

vector<vector<string>> solveNQueens_dfs(int n){
    vector<int> placed(n, 0);
    vector<vector<string>> result;
    
    stack<NQueenNode> path;
    path.push({0, nullptr});
    
    while (!path.empty()) {
        auto &cur = path.top();
        
        //1. 做选择。place代表当前行的皇后放在第几列上
        int place = 0;
        if (cur.forbidden == nullptr) {
            
            //皇后不能在同行同列和斜线上，由之前皇后的放置情况确定当前行禁止的位置
            bool *forbidden = (bool*)malloc(sizeof(bool)*n);
            memset(forbidden, 0, sizeof(sizeof(bool)*n));
            
            for (int i = 0; i<cur.idx; i++) {
                forbidden[placed[i]] = true;
                int left = placed[i]-(cur.idx-i), right = placed[i]+(cur.idx-i);
                if (left >= 0) {
                    forbidden[left] = true;
                }
                if (right < n) {
                    forbidden[right] = true;
                }
            }
            
            cur.forbidden = forbidden;
        }else{
            place = placed[cur.idx]+1;
        }
        
        //有些选择是直接跳到下一条路就可以了，这里还需要越过许多障碍，就是那些不能放皇后的地方
        while (place < n && cur.forbidden[place]) {
            place++;
        }
        
        //2. 只有两种情况时回溯：1. 没有下一步的路了 2.得到解了
        if (place == n) { //没有进一步的选择了
            path.pop();
        }else{
            placed[cur.idx] = place;
            if (cur.idx == n-1) {  //得到解了
                vector<string> ans(n, string(n, '.'));
                for (int i = 0; i<n; i++) {
                    ans[i][placed[i]] = 'Q';
                }
                result.push_back(ans);
                path.pop();
            }else{
                path.push({cur.idx+1, nullptr});
            }
        }
    }
    
    return result;
}

//570. 寻找丢失的数 II
//n其实不用给，可以通过str的长度推测出来，对于一个数n，位数为kn，从1-n所有数长度之和为f(n),则丢失一个数的字符串长度为f(n)-kx,kx是丢失的数的长度，而kx的范围是[1,kn],则字符串范围是[f(n)-kn,f(n)-1]；那么前一个数即n-1的范围是[f(n-1)-kn1, f(n-1)-1],kn1是n-1的长度，而f(n)=f(n-1)+kn,则范围转化为[f(n)-kn-kn1, f(n)-kn-1],所以看得出两个区间是不会重叠的，这也就表示数n和字符串长度存在唯一的关联关系。
int findMissing2(int n, string &str) {
    int count[10];
    memset(count, 0, sizeof(count));
    
    for (int i = 1; i<=min(9, n); i++) {
        count[i]++;
    }
    int digit1 = 1, digit2 = 0;
    for (int i = 10; i<=n; i++) {
        count[digit1]++;
        count[digit2]++;
        digit2++;
        if (digit2==10) {
            digit1++;
            digit2=0;
        }
    }
    
    for (auto &c : str){
        count[c-'0']--;
    }
    
    int miss1=-1, miss2 = -1;
    for (int i = 0; i<10; i++) {
        if (count[i]==1) {
            if (miss1<0) {
                miss1 = i;
            }else{
                miss2 = i;
            }
        }else if (count[i] == 2){
            miss1 = miss2 = i;
        }
    }
    
    int result=0;
    if (miss2<0) {
        result = miss1;
    }else{
        if (miss1 == 0) {
            result = miss2*10+miss1;
        }else{
            result = miss1*10+miss2;
        }
    }
    
    //因为n<=30,所以除了21，都是前面数小
    if (result == 12 && str.find("12") != string::npos) {
        return 21;
    }
    
    return result;
}

TreeNode *copyTree(TreeNode *root, int delta){
    return TreeNode::copy(root, delta);
}

/** 虽然这题用来训练dfs很好，但实际用动态规划更好，因为区间[1,k]和[x+1,x+k]构建二叉树的逻辑是一样的，只是每个对应位置的数叠加x,所以问题就有区间这两个变量变成了长度这一个变量。
 在求出长度为1...k-1的问题后，再求长度为k的问题，选取根节点后，左右区间长度<=k-1,都是已经求出的结果，所以很快就可以得到解。
 奇怪速度并没有加快
 */
vector<TreeNode *> generateTrees_dp(int n){
    if(n==0) return {nullptr};
    vector<TreeNode *> save[n+1];
    save[0] = {nullptr};
    save[1] = {new TreeNode(1)};
    
    double lastCount = 1;
    for (int i = 2; i<=n; i++) {
        long generateTreeCount = 0;
        for (int j = 1; j<=i; j++) {
            
            for (auto &t : save[i-j]){
//                auto r = copyTree(t, j); //格式相同，叠加一个差值
                auto r = t;
                for (auto &l : save[j-1]){
                    generateTreeCount++;
                    auto node = new TreeNode(j);
                    node->left = l;
                    node->right = r;
                    save[i].push_back(node);
                }
            }
        }
        
        printf("[%d] %ld, %.3f\n",i,generateTreeCount,generateTreeCount/lastCount);
        lastCount = generateTreeCount;
    }
    
    return save[n];
}

pair<int, int> houseRobber3_pair(TreeNode * root) {
    if (root == nullptr) {
        return {0,0};
    }
    auto left = houseRobber3_pair(root->left);
    auto right = houseRobber3_pair(root->right);
    
    int involveMax = root->val + left.second + right.second;
    int uninvolveMax = max(left.first, left.second) + max(right.first, right.second);
    
    return {involveMax, uninvolveMax};
}

//535. 打劫房屋 III
int houseRobber3(TreeNode * root) {
    auto p = houseRobber3_pair(root);
    return max(p.first, p.second);
}

//第1个值是整个数里最长路径，第2个值是从root开始到叶节点最长路径，但数值是递增的，第3个值是递减路径
vector<int> longestConsecutive2_pair(TreeNode * root){
    
    if (root == nullptr) {
        return {0, 0, 0};
    }
    
    auto left = longestConsecutive2_pair(root->left);
    auto right = longestConsecutive2_pair(root->right);
    
    int incre1 = 1, decre1 = 1;
    if (root->left) {
        if (root->val == root->left->val+1) { //递减
            decre1 += left[2];
        }
        if (root->val == root->left->val-1) {
            incre1 += left[1];
        }
    }
    int incre2 = 1, decre2 = 1;
    if (root->right) {
        if (root->val == root->right->val+1) { //递减
            decre2 += right[2];
        }
        if (root->val == root->right->val-1) {
            incre2 += right[1];
        }
    }
    
    int leftFirst = decre1+incre2-1;
    int rightFirst = incre1+decre2-1;
    int wholeMaxPath = max(max(leftFirst, rightFirst), max(left[0], right[0]));

    return {wholeMaxPath, max(incre1, incre2), max(decre1, decre2)};
}

//614. 二叉树的最长连续子序列 II
int longestConsecutive2(TreeNode * root) {
    return longestConsecutive2_pair(root)[0];
}

bool luckyNumber(string &n, string &result, int start, int count3, int count5, bool excess){
    if (count3<0 || count5<0) {
        return false;
    }
    if (start == n.size()) {
        return true;
    }
    
    if (!excess && n[start]>'5') {
        return false;
    }
    
    if (excess || n[start]<='3') {
        result[start] = '3';
        if (result[start] > n[start]) {
            excess = true;
        }
        if (luckyNumber(n, result, start+1, count3-1, count5, excess)) {
            return true;
        }
    }
    
    if (excess || n[start]<='5') {
        result[start] = '5';
        if (result[start] > n[start]) {
            excess = true;
        }
        if (luckyNumber(n, result, start+1, count3, count5-1, excess)) {
            return true;
        }
    }
    
    return false;
}

string luckyNumber(string &n) {
    
    int size = (int)n.length()/2;
    if (n.length() & 1 || n.empty()) {
        return string(size+1,'3')+string(size+1, '5');
    }
    
    string result(n.length(), '0');
    if (luckyNumber(n, result, 0, size, size, false)) {
        return result;
    }else{
        return string(size+1,'3')+string(size+1, '5');
    }
}

int lengthOfLongestSubstringKDistinct(string s, int k) {
    //#define charIndex(c) (c>='a'?(c-'a'):(c-'A'+26))
    // write your code here
    int maxlen = 0, uniqueCount = 0;
    int start = 0, end = 0;
    
    int mark[256];
    memset(mark, 0, sizeof(mark));
    
    for(int i=0;i<s.length();i++)
    {
        mark[s[i]]++;
        end = i;
        if (mark[s[i]]==1) {
            uniqueCount++;
        }
        
        while(uniqueCount>k)
        {
            mark[s[start]]--;
            if(mark[s[start]]==0){
                uniqueCount--;
            }
            start++;
        }
        
        //        if (end-start+1>maxlen) {
        //            cout<<s.substr(start,end-start+1)<<endl;
        //        }
        maxlen = max(maxlen, end-start+1);
    }
    return maxlen;
}

int pokeMaster(vector<int> &cards) {
    int counts[10];
    memset(counts, 0, sizeof(counts));
    for (auto &n : cards){
        counts[n]++;
    }
    
    int count = 0;
    while (1) {
        int start = 0, end = 0;
        int max1 = 0, maxStart = 0;
        for (int i = 1; i<10; i++) {
            if (counts[i] > 0) {
                end++;
                if (end-start > max1) {
                    max1 = end-start;
                    maxStart = start;
                }
            }else{
                start = end = i;
            }
        }
        
        int max2 = 0, maxIdx = 0;
        for (int i = 1; i<10; i++) {
            if (counts[i]>max2) {
                max2 = counts[i];
                maxIdx = i;
            }
        }
        
        if (max2 == 0) {
            break;
        }
        
        if (max1 >= 5 && max1 > max2) {
            for (int i = maxStart+1; i<=maxStart+max1; i++) {
                counts[i]--;
            }
        }else{
            counts[maxIdx] = 0;
        }
        
        count++;
    }
    
    return count;
}

class TopK {
    struct wordRecord{
        string word;
        int time;
    };
    static inline int wordRecordComp(wordRecord* &w1, wordRecord* &w2){
        if (w1->time < w2->time) {
            return -1;
        }else if (w1->time > w2->time){
            return 1;
        }else{
            return w2->word.compare(w1->word);
        }
    }
    
    TFDataStruct::heap<wordRecord*> *minHeap = nullptr;
    int size = 0;
    unordered_map<string, wordRecord*> wordMap;
public:
    
    
    TopK(int k) {
        size = k;
        minHeap = new TFDataStruct::heap<wordRecord*>(wordRecordComp, k);
    }
    
    void add(string &word) {
        
        bool inHeap = false;
        wordRecord *wr = nullptr;
        if (wordMap.find(word) == wordMap.end()) {
            wr = new wordRecord;
            wr->word = word;
            wr->time = 1;
            wordMap[word] = wr;
        }else{
            wr = wordMap[word];
            
            if (wr->time > minHeap->getTop()->time) {
                inHeap = true;
            }else if (wr->time == minHeap->getTop()->time){
                inHeap = minHeap->exist(wr);
            }
            wr->time++;
        }
        
        if (inHeap) {
            minHeap->update(wr);
        }else{
            if (minHeap->isFull() && !minHeap->isEmpty()) {
                auto top = minHeap->getTop();
                if (wordRecordComp(wr, top)>0) {
                    minHeap->replaceTop(wr);
                }
            }else{
                minHeap->append(wr);
            }
        }
    }
    
    vector<string> topk() {
        auto limit = min((int)minHeap->getValidSize(), size);
        vector<string> result(limit, "");
        vector<wordRecord *> saveNodes(limit, nullptr);
        
        for (int i = 0; i<limit; i++) {
            saveNodes[limit-i-1] = minHeap->getTop();
            result[limit-i-1] = minHeap->popTop()->word;
        }
        for (auto val : saveNodes){
            minHeap->append(val);
        }
        return result;
    }
};

bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
    
    typedef TFDataStruct::DirectedGraph<int> GraphType;
    GraphType *graph = new GraphType();
    
    for (int i = 0; i<numCourses; i++) {
        graph->allNodes.push_back(GraphType::NodeType(i));
    }
    
    for (auto &pair : prerequisites){
        graph->allNodes[pair.first].adjNodes.push_back(&graph->allNodes[pair.second]);
    }
    return !graph->isCyclic();
}

class Connection {
    public:
    string city1, city2;
    int cost;
//    Connection(string& city1, string& city2, int cost) {
//        this->city1 = city1;
//        this->city2 = city2;
//        this->cost = cost;
//    }
};

bool connectComp(Connection &con1, Connection &con2){
    int result = con1.cost-con2.cost;
    if (result<0) {
        return true;
    }else if (result>0){
        return false;
    }

    result = con1.city1.compare(con2.city1);
    if (result<0) {
        return true;
    }else if (result>0){
        return false;
    }

    result = con1.city2.compare(con2.city2);
    return result<=0;
}

vector<Connection> lowestCost(vector<Connection>& connections) {
    typedef TFDataStruct::UndirectedGraph<string> GraphType;
    GraphType graph;
    
    map<string, int> cityIdx;
    for (auto &con : connections){
        
        
        int edgeIdx1 = -1;
        if (cityIdx.find(con.city1) == cityIdx.end()) {
            graph.allNodes.push_back(GraphType::NodeType(con.city1));
            edgeIdx1 = cityIdx[con.city1] = (int)graph.allNodes.size()-1;
        }else{
            edgeIdx1 = cityIdx[con.city1];
        }
        
        int edgeIdx2 = -1;
        if (cityIdx.find(con.city2) == cityIdx.end()) {
            graph.allNodes.push_back(GraphType::NodeType(con.city2));
            edgeIdx2 = cityIdx[con.city2] = (int)graph.allNodes.size()-1;
        }else{
            edgeIdx2 = cityIdx[con.city2];
        }
        
        
        graph.allNodes[edgeIdx1].edges.push_back({edgeIdx2, con.cost});
        graph.allNodes[edgeIdx2].edges.push_back({edgeIdx1, con.cost});
    }
    
    //默认第一个节点为根，根没有前置节点，不处理
    auto closest = graph.lowestCost_prim();
    vector<Connection> result;
    for (int i = 1; i<closest.size(); i++){
        auto &edge = closest[i];

        string &val1 = graph.allNodes[i].val;
        string &val2 = graph.allNodes[edge.other].val;
        bool inc = val1.compare(val2)<0;
        string minSide = min(val1, val2);
        string maxSide = max(val1, val2);
        result.push_back({inc?val1:val2, inc?val2:val1, edge.cost});
    }
    
    sort(result.begin(), result.end(), connectComp);
    
    return result;
}

vector<Connection> lowestCost_mat(vector<Connection>& connections) {
    typedef TFDataStruct::UndirectedGraph<string> GraphType;
    GraphType graph;
    graph.initMatrix((int)connections.size());
    
    map<string, int> cityIdx;
    for (auto &con : connections){
        int edgeIdx1 = -1;
        if (cityIdx.find(con.city1) == cityIdx.end()) {
            graph.allNodes.push_back(GraphType::NodeType(con.city1));
            edgeIdx1 = cityIdx[con.city1] = (int)graph.allNodes.size()-1;
        }else{
            edgeIdx1 = cityIdx[con.city1];
        }
        
        int edgeIdx2 = -1;
        if (cityIdx.find(con.city2) == cityIdx.end()) {
            graph.allNodes.push_back(GraphType::NodeType(con.city2));
            edgeIdx2 = cityIdx[con.city2] = (int)graph.allNodes.size()-1;
        }else{
            edgeIdx2 = cityIdx[con.city2];
        }
        
        graph.matrix[edgeIdx1][edgeIdx2] = con.cost;
        graph.matrix[edgeIdx2][edgeIdx1] = con.cost;
    }
    
    auto edges = graph.lowestCost_kruskal();
    vector<Connection> result;
    for (auto &edge : edges) {
        
        string &val1 = graph.allNodes[edge.first].val;
        string &val2 = graph.allNodes[edge.second].val;
        bool inc = val1.compare(val2)<0;
        string minSide = min(val1, val2);
        string maxSide = max(val1, val2);
        result.push_back({inc?val1:val2, inc?val2:val1, edge.cost});
    }
    
    sort(result.begin(), result.end(), connectComp);
    
    return result;
}

#define LRUCache(c) LRUCache cache(c);
int main(int argc, const char * argv[]) {
    
    uint64_t start = mach_absolute_time();
    
    
    vector<Connection> connections;
    
    string path = "/Users/apple/Desktop/connections_data";
    readFile(path, [&connections](string &line){
        
        Connection con;
        int last = 1, idx = 0;
        for (int i = 0; i< line.size(); i++){
            char c = line[i];
            if (c == ',' || c == ']') {
                if (idx == 0) {
                    con.city1 = line.substr(last+1, i-last-2);
                }else if (idx == 1){
                    con.city2 = line.substr(last+1, i-last-2);
                }else{
                    int num = 0, digit = 1;
                    for (int k = i-1; k>=last; k--) {
                        num += (line[k]-'0')*digit;
                        digit *= 10;
                    }
                    con.cost = num;
                }
                
                idx++;
                last = i+1;
            }
        }
        
        connections.push_back(con);
    });
    
//    for (auto &con : connections){
//        cout<<con.city1<<","<<con.city2<<","<<con.cost<<endl;
//    }
    
    auto result = lowestCost(connections);
    for (auto &con : result){
        cout<<con.city1<<","<<con.city2<<","<<con.cost<<endl;
    }
    
    uint64_t duration = mach_absolute_time() - start;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
    printf("exe time: %.1f ms\n",time);
//    printf("%d \n",result);
//    printVectorOneLine(result);
    
//    canGameOver(nums, 500549646);
}
