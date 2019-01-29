//
//  page2.hpp
//  algorithm
//
//  Created by shiwei on 2018/5/22.
//

#include <stdio.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <stack>
#include <iostream>
#include "heap.hpp"
#include <mach/mach_time.h>
#include <unordered_set>

#include "TFSort.h"
#include "MinStack.hpp"
#include "BinaryTree.hpp"
#include "CommonStructs.hpp"

#ifndef page2_h
#define page2_h

int maxSquare(vector<vector<int>> &matrix) {
    if (matrix.empty()) {
        return 0;
    }
    
    int maxRes = 0;
    
    for (int i = 1; i < matrix.size(); i++) {
        vector<int> &row = matrix[i];
        for (int j = 0; j<row.size(); j++) {
            if (row[j]) {
                row[j] = min(min(row[j-1], matrix[i-1][j]), matrix[i-1][j-1])+1;
                
                maxRes = max(maxRes, row[j]);
            }
        }
    }
    
    return maxRes*maxRes;
}

struct WordTime{
    string word;
    int time;
};

//堆是按最小队逻辑实现的，返回负数，会认为前一个小，把前一个放到上面
static int wordTimeCompare(WordTime w1, WordTime w2){
    if (w1.time < w2.time) {
        return TFDataStruct::TFHeapComparePreDown;
    }else if (w1.time > w2.time){
        return TFDataStruct::TFHeapComparePreUp;
    }else{
        return w1.word.compare(w2.word);
    }
}

vector<string> topKFrequentWords(vector<string> &words, int k) {
    TFDataStruct::heap<WordTime> wordHeap(wordTimeCompare, words.size());
    
    map<string, int> wordTimes;
    for (auto iter = words.begin(); iter != words.end(); iter++) {
        wordTimes[*iter]++;
    }
    
    for (auto pair : wordTimes){
        wordHeap.append((WordTime){pair.first, pair.second});
    }
    
    vector<string> result;
    for (int i = 0; i<k; i++) {
        result.push_back(wordHeap.popTop().word);
    }
    
    return result;
}

void DPSChangeO(vector<vector<char>> &board, int x, int y){
    vector<Point> forks;
    
    board[x][y] = '*';
    
    while (1) {
        
        if (x > 0 && board[x-1][y] == 'O') {
            board[x-1][y] = '*';
            forks.push_back((Point){x-1,y});
        }
        if (x < board.size()-1 && board[x+1][y] == 'O') {
            board[x+1][y] = '*';
            forks.push_back((Point){x+1,y});
        }
        if (y > 0 && board[x][y-1] == 'O') {
            board[x][y-1] = '*';
            forks.push_back((Point){x,y-1});
        }
        if (y < board.front().size()-1 && board[x][y+1] == 'O') {
            board[x][y+1] = '*';
            forks.push_back((Point){x,y+1});
        }
        
        if (forks.empty()) {
            break;
        }
        x = forks.back().x;
        y = forks.back().y;
        forks.pop_back();
    }
}

void surroundedRegions(vector<vector<char>> &board) {
    if (board.empty()) {
        return;
    }
    
    int width = board.size();
    int height = board.front().size();
    
    for (int i = 0; i<width; i++) {
        if (board[i][0] == 'O') {
            DPSChangeO(board, i, 0);
        }
        if (board[i][height-1] == 'O') {
            DPSChangeO(board, i, height-1);
        }
    }
    
    for (int j = 1; j<height-1; j++) {
        if (board[0][j] == 'O') {
            DPSChangeO(board, 0, j);
        }
        if (board[width-1][j] == 'O') {
            DPSChangeO(board, width-1, j);
        }
    }
    
    for (int i = 0; i<width; i++) {
        for (int j = 0; j<height; j++) {
            if (board[i][j] == '*') {
                board[i][j] = 'O';
            }else if (board[i][j] == 'O'){
                board[i][j] = 'X';
            }
        }
    }
}

//508. 摆动排序
//这个等于处理不了，但可以应对508
void wiggleSort(vector<int> &nums) {
    if (nums.size() < 2) {
        return;
    }
    
    bool greater = true;
    for (int i = 1; i<nums.size(); i++) {
        if (nums[i] == nums[i-1]) {
            
        }
        if (greater != nums[i] > nums[i-1]) {
            int temp = nums[i-1];
            nums[i-1] = nums[i];
            nums[i] = temp;
        }
        
        greater = !greater;
    }
}

//507. 摆动排序 II
void wiggleSort2(vector<int>& nums) {
    int n = nums.size();
    
    // Find a median.
    auto midptr = nums.begin() + n / 2;
    nth_element(nums.begin(), midptr, nums.end());
    int mid = *midptr;
    
    /*      i       A(i)
     0       1
     中间是所有波峰
     n/2-1   最后一个波峰
     n/2     0
     中间是所有波谷
     n-1     最后一个波谷
     */
    //所以j和k所踩得点连起来是所有的点，而且是波峰在前，波谷在后。最后就分成了3段，[A(0), A(i-1)]是大于，[A(i), A(j-1)]是等于，[A(k+1),A(n-1)]是小于
    //因为中位数的性质，大于和小于的个数是一样的，或者大于多一个，那么从中间切开，前面是波峰，后面是波谷，那么波峰就是大于+等于，波谷是等于+小于。除非等于的个数超过一半，两头的等于碰到一起才无解。
#define A(i) nums[(1+2*(i)) % (n|1)]
    
    
    //思路是：在偶数位，也就是波峰位置，把大于中位数的放左边，等于的放右边。j和i的操作
    //在奇数为，就是波谷位置，把低于中位数的调到这里。这时j和k的操作
    //最后一定会是在偶数为，左边为大于，右边为等于，或者没有等于，在奇数为，都是小于。问题在于奇数位怎么保证一定遍历完？
    
    int i = 0, j = 0, k = n - 1;
    while (j <= k) {
        if (A(j) > mid)
            swap(A(i++), A(j++));
        else if (A(j) < mid)
            swap(A(j), A(k--));
        else
            j++;
    }
}

ListNode * swapNodes(ListNode * head, int v1, int v2) {
    ListNode *pre1 = nullptr, *pre2 = nullptr, *find1 = nullptr, *find2 = nullptr;
    
    ListNode *cur = head, *pre = nullptr;
    while (cur) {
        if (v1 == cur->val) {
            find1 = cur;
            pre1 = pre;
            if (find2) {
                break;
            }
        }
        if (v2 == cur->val) {
            find2 = cur;
            pre2 = pre;
            if (find1) {
                break;
            }
        }
        pre = cur;
        cur = cur->next;
    }
    
    if (find1 == nullptr || find2 == nullptr) {
        return head;
    }
    
    
    if (find1->next == find2) {
        auto next2 = find2->next;
        if (pre1) {
            pre1->next = find2;
        }else{
            head = find2;
        }
        find2->next = find1;
        find1->next = next2;
    }else if (find2->next == find1){
        auto next1 = find1->next;
        if (pre2) {
            pre2->next = find1;
        }else{
            head = find1;
        }
        find1->next = find2;
        find2->next = next1;
    }else{
        auto next1 = find1->next;
        auto next2 = find2->next;
        if (pre1) {
            pre1->next = find2;
        }else{
            head = find2;
        }
        find2->next = next1;
        if (pre2) {
            pre2->next = find1;
        }else{
            head = find1;
        }
        find1->next = next2;
    }
    return head;
}

//512. 解码方法
//简单而典型的动态规划体，k依赖于k-1和k-2,所以空间复杂度可以压缩到O(1),时间复杂度为O(n)
int numDecodings(string &s) {
    if (s.empty()) {
        return 0;
    }
    
    //分别代表k-1和k-2的情况
    int lastDigit = s.back()-'0';
    int num1 = lastDigit==0?0:1, num2 = 1;
    for (int i = s.length()-2; i>=0; i--) {
        int curDigit = s[i]-'0';
        int curNum = 0;
        if (curDigit != 0) {
            curNum += num1;
            
            if (curDigit*10+lastDigit <= 26) {
                curNum += num2;
            }
        }
        
        num2 = num1;
        num1 = curNum;
        lastDigit = curDigit;
    }
    
    return num1;
}

//513. 完美平方
//这也是一个动态规划的题，但是k是依赖于多个前面的，而且依赖规则是变化的
int numSquares(int n) {
    int res[n+1];
    res[0] = 0;
    res[1] = 1;
    
    int sq = 1, sqPow = 4;
    for (int i = 2; i<=n; i++) {
        if (i == sqPow) {
            res[i] = 1;
            sq++;
            sqPow = (sq+1)*(sq+1);
        }else{
            res[i] = INT_MAX;
            for (int j = sq; j>=1; j--) {
                res[i] = min(res[i], res[i-j*j]+1);
            }
        }
    }
    
    return res[n];
}

int nthSuperUglyNumber(int n, vector<int> &primes) {
    
    vector<int> uglyNums = {1};
    int primesIndex[primes.size()];
    memset(primesIndex, 0, sizeof(primesIndex));
    
    //TODO:用一个最小堆来维持可以把复杂度从O(nk)降到O(nlgk),k较大时会效果明显
    for (int i = 1; i<n; i++) {
        int next = INT_MAX;
        int minPrime = 0;
        for (int j = 0; j<primes.size(); j++) {
            auto cand = primes[j]*uglyNums[primesIndex[j]];
            
            //上一把有重复的
            if (cand == uglyNums.back()) {
                primesIndex[j]++;
                cand = primes[j]*uglyNums[primesIndex[j]];
            }
            
            if (cand < next) {
                next = cand;
                minPrime = j;
            }
        }
        
        printf("%d ",minPrime);
        
        primesIndex[minPrime]++;
        uglyNums.push_back(next);
    }
    
    return uglyNums[n-1];
}

class ZigzagIterator {
public:
    
    vector<int> v1;
    vector<int> v2;
    
    bool first = true;
    int idx1 = 0;
    int idx2 = 0;
    
    ZigzagIterator(vector<int>& v1, vector<int>& v2) {
        this->v1 = v1;
        this->v2 = v2;
        
        if (v1.empty()) {
            first = false;
        }
    }
    
    /*
     * @return: An integer
     */
    int next() {
        
        if (first) {
            first = idx2 == v2.size();
            return v1[idx1++];
        }else{
            first = idx1 != v1.size();
            return v2[idx2++];
        }
    }
    
    /*
     * @return: True if has next
     */
    bool hasNext() {
        return idx1 != v1.size() || idx2 != v2.size();
    }
};

int maxKilledEnemies(vector<vector<char>> &grid) {
    if (grid.empty()) {
        return 0;
    }
    
    int width = grid.size();
    int height = grid.front().size();
    int kills[width*height];
    memset(kills, 0, sizeof(kills));
    
    for (int i = 0; i<width; i++) {
        int curE = 0;
        for (int j = 0; j<height; j++) {
            
            int index = i*height+j;
            if (grid[i][j] == 'W'){
                curE = 0;
            }else if (grid[i][j] == 'E') {
                curE++;
            }else if (grid[i][j] == '0'){
                kills[index] = curE;
            }
        }
    }
    
    printf("%d \n",kills[5]);
    
    for (int i = 0; i<width; i++) {
        int curE = 0;
        for (int j = height-1; j>=0; j--) {
            
            int index = i*height+j;
            if (grid[i][j] == 'W'){
                curE = 0;
            }else if (grid[i][j] == 'E') {
                curE++;
            }else if (grid[i][j] == '0'){
                kills[index] += curE;
            }
        }
    }
    
    printf("%d \n",kills[5]);
    
    for (int j = 0; j<height; j++) {
        
        int curE = 0;
        
        for (int i = 0; i<width; i++) {
            int index = i*height+j;
            if (grid[i][j] == 'W'){
                curE = 0;
            }else if (grid[i][j] == 'E') {
                curE++;
            }else if (grid[i][j] == '0'){
                kills[index] += curE;
            }
        }
    }
    
    printf("%d \n",kills[5]);
    
    int maxE = 0;
    for (int j = 0; j<height; j++) {
        
        int curE = 0;
        
        for (int i = width-1; i>=0; i--) {
            int index = i*height+j;
            if (grid[i][j] == 'W'){
                curE = 0;
            }else if (grid[i][j] == 'E') {
                curE++;
            }else if (grid[i][j] == '0'){
                maxE = max(maxE, kills[index]+curE);
            }
        }
    }
    
    return maxE;
}

int findMissing2(int n, string &str) {
    int count[n+1];
    memset(count, 0, sizeof(count));
    count[0] = 1;
    
    for (int i = 0; i<str.length(); i++) {
        short num = str[i]-'0';
        if (num+10 <= n && !count[num+10]){
            count[num+10] = 1;
            count[1]--;
        }else if(num+20 <= n && !count[num+20]){
            count[num+20] = 1;
            count[2]--;
        }else{
            if (num == 0) {
                count[30] = 1;
                count[3]--;
            }
            count[num]++;
        }
    }
    
    short miss1 = n+1, miss2 = n+1;
    for (int i = 1; i<=n; i++) {
        printf("[%d]%d ",i,count[i]);
        if (count[i] == 0) {
            if (miss1 > n) {
                miss1 = i;
            }else{
                miss2 = i;
                break;
            }
        }else if (count[i] < 0){
            return i*11;
        }
    }
    
    if (miss1 > n) {
        return 30;
    }
    
    int result = miss1*10+miss2;
    if (miss2 > n) {
        result = miss1;
    }
    if (result == 12 && str.find("12") != string::npos) {
        return 21;
    }
    
    return result;
}

vector<Interval> mergeKSortedIntervalLists(vector<vector<Interval>> &intervals) {
    vector<Interval> result;
    int nextIndexes[intervals.size()];
    memset(nextIndexes, 0, sizeof(nextIndexes));
    
    Interval next = Interval(INT_MAX, INT_MAX);
    
    bool finished = false;
    while (!finished) {
        
        finished = true;
        int minList = 0;
        for (int i = 0; i < intervals.size(); i++) {
            int index = nextIndexes[i];
            vector<Interval> &inters = intervals[i];
            if (index < inters.size()) {
                finished = false;
                if (inters[index].start < next.start) {
                    next = inters[index];
                    minList = i;
                }
            }
        }
        
        if (finished) {
            break;
        }
        
        if (result.empty()) {
            result.push_back(next);
        }else{
            Interval &last = result.back();
            if (last.end <= next.start) {
                result.push_back(next);
            }else if (last.end < next.end){
                last.end = next.end;
            }
        }
        
        nextIndexes[minList]++;
        next = Interval(INT_MAX, INT_MAX);
    }
    return result;
}

//588. 划分和相等的子集
//转化为选取一个子集，和为总体的一半，总数是确定的，一半也就是确定的。对每个元素，有加入和不加入两种选择，这个问题就转成了背包问题
//跟给定一个目标值，求一个子集使得和为目标值额题目是一样的
bool canPartition(vector<int> &nums) {
    
    int sum = 0;
    for (int i = 0; i<nums.size(); i++) {
        sum += nums[i];
    }
    
    if (sum & 1) { //奇数
        return false;
    }
    
    int target = sum/2;
    bool sumPossible[target];
    memset(sumPossible, 0, sizeof(sumPossible));
    sumPossible[0] = true;
    
    for (int i = 0; i<nums.size(); i++) {
        int cur = nums[i];
        if (sumPossible[target-cur]) {
            return true;
        }
        for (int j = target - cur-1; j>=0; j--) {
            sumPossible[j+nums[i]] |= sumPossible[j];
        }
    }
    
    return false;
}

//601. 摊平二维向量
class Vector2D {
public:
    vector<vector<int>> *orgData = nullptr;
    vector<vector<int>>::iterator iter1;
    vector<int>::iterator iter2;
    
    Vector2D(vector<vector<int>>& vec2d) {
        orgData = &vec2d;
        iter1 = vec2d.begin();
        if (iter1 != vec2d.end()) {
            iter2 = iter1->begin();
            moveToNext();
        }
    }
    
    void moveToNext(){
        while (iter2 == iter1->end()) {
            iter1++;
            if (iter1 == orgData->end()) {
                break;
            }
            iter2 = iter1->begin();
        }
    }
    
    int next() {
        auto value = *iter2++;
        moveToNext();
        return value;
    }
    
    bool hasNext() {
        return iter1 != orgData->end();
    }
};

//603. 最大整除子集
//这个错误解法保留，这个是动态规划的思想，但是实际上k情况需要前面所有k-1的，代价有点大。
//或许可以使用图的思想，每个相邻且就有整除关系的数之间连线，这样建立一个完整的图，然后使用深度搜索，找到路线最长的那一条路线。
//最短路径是经典问题，但最长和最短性质是一样的
vector<int> largestDivisibleSubset2(vector<int> &nums) {
    if (nums.empty()) {
        return {};
    }
    
    sort(nums.begin(), nums.end());
    
    vector<vector<int>> divs;
    for (int i = 0; i < nums.size(); i++) {
        int cur = nums[i];
        
        bool find = false;
        for (int i = 0; i<divs.size(); i++){
            vector<int> &subset = divs[i];
            if (cur%subset.back() == 0) {
                find = true;
                subset.push_back(cur);
            }
        }
        
        if (!find) {
            divs.push_back({cur});
        }
    }
    
    printTwoDVector(divs);
    
    vector<int> *maxSubSet = nullptr;
    int maxLen = 0;
    for (int i = 0; i<divs.size(); i++){
        vector<int> &subset = divs[i];
        if (subset.size() > maxLen) {
            maxLen = (int)subset.size();
            maxSubSet = &subset;
        }
    }
    
    return *maxSubSet;
}

struct DivideNode {
    int orgPos;
    int val = 0;
    DivideNode *pre = nullptr;
    int maxLen = 1;
    
    DivideNode(int orgPos, int val):orgPos(orgPos),val(val){};
};

static bool DivideNodeComp1(const DivideNode &n1, const DivideNode &n2){
    return n1.val < n2.val;
}

static bool DivideNodeComp2(const DivideNode *n1, const DivideNode *n2){
    return n1->orgPos < n2->orgPos;
}

vector<int> largestDivisibleSubset(vector<int> &nums) {
    
    vector<DivideNode> nodes;
    for (int i = 0; i < nums.size(); i++) {
        nodes.push_back(DivideNode(i, nums[i]));
    }
    
    sort(nodes.begin(), nodes.end(), DivideNodeComp1);
    
    DivideNode *maxLenNode = nullptr;
    int maxLen = 0;
    for (int i = 1; i<nums.size(); i++) {
        for (int j = i-1; j>=0; j--) {
            if (nodes[j].maxLen+1 > nodes[i].maxLen && (nodes[i].val % nodes[j].val) == 0) {
                nodes[i].maxLen = nodes[j].maxLen+1;
                nodes[i].pre = &nodes[j];
            }
        }
        
        if (nodes[i].maxLen > maxLen) {
            maxLen = nodes[i].maxLen;
            maxLenNode = &nodes[i];
        }
    }
    
    vector<DivideNode *> path;
    DivideNode *cur = maxLenNode;
    while (cur) {
        path.push_back(cur);
        cur = cur->pre;
    }
    
    sort(path.begin(), path.end(), DivideNodeComp2);
    
    vector<int> result;
    for (auto &node : path){
        result.push_back(node->val);
    }
    
    return result;
}

//614. 二叉树的最长连续子序列 II
//任何一个路径，一定有一个最高级的节点，而这整个路径可以看成这个点的两条子路径合成的，而且一定是包含两边的，只要两个节点都有。所以可以以定节点为元素把情况划分比较。
// ** 一个复杂问题的解决，依赖于将问题划分为更低级但同性质的问题，最好的是这些同性质问题直接有着更快速的联系 **
//返回包含在root这棵树里的最长路径，并不经过root，maxLen是最长的以root为起点的路径
int longestConsecutive2(TreeNode * root, int *maxLen){
    int left1 = 0, left2 = 0, right1 = 0, right2 = 0;
    if (root->left) {
        left1 = longestConsecutive2(root->left, &left2);
    }
    if (root->right) {
        right1 = longestConsecutive2(root->right, &right2);
    }
    
    *maxLen = max(left2, right2)+1;
    int curMaxPath = left2+right2+1;
    return max(max(left1, right1), curMaxPath);
}

int longestConsecutive2(TreeNode * root) {
    int x = 0;
    return longestConsecutive2(root, &x);
}

/*
 假设有两组，k1长度的均值为A1，k2长度的均值为A2，且k1>k2,A1<=A2,即长度大的均值小。
 然后在它们前面都加上一节数据，h个数字，总值m，那么前一组均值变为(k1A1+m)/(k1+h),后一组为(k2A2+m)/(k2+h).
 两者相减值<=0，且设A2=A1+t,m=A1+p,得到(k2-k1)(A1-hA1+p)-k2t(k1+h)<=0;A1-hA1+p=m-hA1,如果m/h>=A1,则上面成立。
 m/h的意义是新增端的均值,那么如果之前新增的均值更大，那么后一组均值还是小。如果之前的新增一段的均值更小，那么两组的均值较当前都会变小，那么就不会取新增后的结果为解，而之前的结果是后一组均值小。所以只要满足k1>k2,A1<A2这种条件，那么第2组肯定淘汰。如果把均值和长度的图做出来，就是低于前一个波谷的点都要切除。也就是一定保持递增的状态。
 推理2:对比递增的两个，A1>A2,k1>k2,则比较结果C=k2t(k1+h)-(k1-k2)(m-A1h)>0,则在m/h<A1时必定成立。当h=1时，大于当前数curNum的不会变的比小于curNum的小，也就是被curNum分割的两端的关系是不会变的。而且对于后一段自身，每两个点之间也满足上面额关系，所以他们之间的关系也不变。
 简单理解：后面的均值会变成[curNum,orgNum2]之间的一个数，前面的均值会变成[org1,curNum]之间的一个数，后面的肯定比前面大。对于后面本身，靠前的减少的多，靠后的减少的少，即小的减少的更多，那么相对关系不会变。
 */

template <class T>
struct TPListNode {
public:
    T val;
    TPListNode *next = nullptr;
    TPListNode(T val):val(val){};
};

struct AverageNode {
    double val;
    double len;
};

double maxAverage(vector<int> &nums, int k) {
    TPListNode<AverageNode> *averHead;
    
    double average = 0;
    for (auto i = nums.size()-k; i<nums.size(); i++) {
        average += nums[i]/(double)k;
    }
    //    printf("%.4f\n",average);
    averHead = new TPListNode<AverageNode>((AverageNode){average, (double)k});
    
    double maxResult = average;
    for (long start = nums.size()-k-1; start>=0; start--) {
        
        double curVal = averHead->val.val+(nums[start]-nums[start+k])/(double)k;
        auto curKAverage = new TPListNode<AverageNode>((AverageNode){curVal, (double)k});
        
        curKAverage->next = averHead;
        averHead = curKAverage;
        
        //把averages变为绝对递增:一个跑酷游戏
        auto peak = averHead;
        auto cur = averHead->next;
        bool hasHigher = false;
        
        while (cur) {
            cur->val.len++;
            double delta = (nums[start]-cur->val.val)/cur->val.len;
            cur->val.val += delta; //新值，增加了当前的节点之后均值变化
            
            if (delta < 0) {
                hasHigher = true;
                if (peak->next != cur) {
                    peak->next = cur;
                }
                cur = cur->next;
                break;
            }
            
            if (cur->val.val > peak->val.val) {
                if (peak->next != cur) {
                    peak->next = cur;
                }
                peak = cur;
                cur = cur->next;
            }else{
                auto drop = cur;
                cur = cur->next;
                free(drop);
            }
        }
        
        if (hasHigher) {
            while (cur) {
                cur->val.len++;
                double delta = (nums[start]-cur->val.val)/cur->val.len;
                cur->val.val += delta;
                cur = cur->next;
            }
        }else{
            peak->next = nullptr;
            maxResult = max(peak->val.val, maxResult);
        }
    }
    return maxResult;
}

//633. 寻找重复的数
int findDuplicate(vector<int> &nums) {
    for (int i = 0; i<nums.size(); i++) {
        int cur = nums[i];
        if (cur == i) {
            continue;
        }
        nums[i] = 0; //空出来
        while (cur > 0) {
            int next = nums[cur];
            if (next == cur) {
                return cur;
            }
            nums[cur] = cur;
            cur = next;
        }
    }
    
    return -1;
}

bool isOneEditDistance(string &s, string &t) {
    
    if (s.empty() && t.empty()) {
        return false;
    }
    if (s.length() + t.length() == 1) {
        return true;
    }
    
    int lenDiff = s.length() - t.length();
    if (abs(lenDiff) > 1) {
        return false;
    }
    
    bool metChange = false;
    int i = 0, j = 0;
    while (i<s.length() && j<t.length()) {
        if (s[i] == t[j]) {
            i++;
            j++;
        }else{
            if (!metChange) {
                metChange = true;
                if (lenDiff == 0) {
                    i++;
                    j++;
                }else if (lenDiff == 1){
                    i++;
                }else{
                    j++;
                }
            }else{
                return false;
            }
        }
    }
    
    return metChange || (lenDiff != 0);
}

inline string genRangeStr(int fr, int to){
    if (fr == to) {
        return to_string(fr);
    }else{
        return to_string(fr)+"->"+to_string(to);
    }
}

vector<string> findMissingRanges(vector<int> &nums, int lower, int upper) {
    
    if (nums.empty()) {
        return {genRangeStr(lower, upper)};
    }
    
    vector<string> result;
    int last = nums.front();
    if (last != lower) {
        result.push_back(genRangeStr(lower, last-1));
    }
    
    for (auto iter = nums.begin()+1; iter != nums.end(); iter++) {
        
        if (*iter == last) {
            continue;
        }
        if (*iter != last+1) {
            result.push_back(genRangeStr(last+1, *iter-1));
        }
        last = *iter;
    }
    
    if (last != upper) {
        result.push_back(genRangeStr(last+1, upper));
    }
    
    return result;
}


bool knows(int a, int b){
    if (a == 0 && b == 3) {
        return true;
    }else if (a == 1 && b == 4){
        return false;
    }else if (a == 2 && b == 5){
        return true;
    }else if (a == 3 && b == 5){
        return true;
    }else if (a == 5 && b == 1){
        return true;
    }
    return a > b;
}

int findCelebrity(int n) {
    int p[n];
    for (int i = 0; i<n; i++) {
        p[i] = i;
    }
    int count = n, step = (count+1)/2;
    while (count > 1) {
        for (int i = 0; i<count/2; i++) {
            if (knows(p[i], p[i+step])) {
                p[i] = p[i+step];
            }
        }
        count = step;
        step = (count+1)/2;
    }
    
    int candicate = p[0];
    for (int i = 0; i<n; i++) {
        if (i != candicate && knows(candicate, i)) {
            return false;
        }
    }
    
    return true;
}

//649. Binary Tree Upside Down
//这题跟题目描述不符：1. 并不是所有由节点都是叶节点或空 2.我们需要把能够逆转的节点提取出来，只旋转它们
//哪些可以逆转：因为逆转后，左节点变成了父节点，所以新的根是最左边的节点。只有从旧root到新root的路线上的所有做节点才是可以做逆转的，因为其他的节点做了处理后，沿着它的左节点方向是到不了新root的，那么就会有多个父节点跑出来了。
TreeNode * upsideDownBinaryTree(TreeNode * root) {
    if (root == nullptr) {
        return nullptr;
    }
    
    TreeNode *cur = root, *left = cur->left, *right = cur->right;
    while (left) {
        
        auto nextLeft = left->left;
        left->left = right;
        
        right = left->right;
        left->right = cur;
        
        cur = left;
        left = nextLeft;
    }
    
    if (cur != root) {
        root->left = nullptr;
        root->right = nullptr;
    }
    
    return cur;
}

int markLeaveDis(TreeNode *root, map<TreeNode *, int> &marks){
    
    int mark = 0;
    if (root->left) {
        mark = markLeaveDis(root->left, marks)+1;
    }
    if (root->right) {
        mark = max(markLeaveDis(root->right, marks)+1, mark);
    }
    
    marks[root] = mark;
    return mark;
}

vector<vector<int>> findLeaves(TreeNode * root) {
    if (root == nullptr) {
        return {};
    }
    map<TreeNode *, int> marks;
    
    markLeaveDis(root, marks);
    
    vector<vector<int>> result;
    TPListNode<TreeNode *> *candicates = new TPListNode<TreeNode *>(nullptr);
    candicates->next = new TPListNode<TreeNode *>(root);
    int mark = marks[root];
    
    while (candicates->next != nullptr) {
        
        vector<int> leavesVal;
        auto pre = candicates;
        auto cur = pre->next;
        while (cur) {
            
            TreeNode *node = cur->val;
            if (marks[node] == mark) {
                leavesVal.push_back(node->val);
                
                pre->next = cur->next;
                free(cur);
                cur = pre->next;
                
                if (node->left) {
                    pre->next = new TPListNode<TreeNode *>(node->left);
                    pre->next->next = cur;
                    
                    pre = pre->next;
                }
                if (node->right) {
                    pre->next = new TPListNode<TreeNode *>(node->right);
                    pre->next->next = cur;
                    
                    pre = pre->next;
                }
                
            }else{
                pre = cur;
                cur = cur->next;
            }
        }
        
        
        result.insert(result.begin(), leavesVal);
        mark--;
    }
    
    return result;
}

//651. 二叉树垂直遍历
//从上往下一层层遍历，每个节点的左边标记-1，右边+1，维持当前的最小和最大标记，根据自身标记和最小标记就可以知道在结果集里的索引。
//索引超出，就扩张结果集
vector<vector<int>> verticalOrder(TreeNode * root) {
    if (root == nullptr) {
        return {};
    }
    
    vector<vector<int>> result;
    auto plane = new vector<pair<TreeNode *, int>>();
    
    int min = 0, max = 0;
    
    plane->push_back({root, 0});
    result.push_back({root->val});
    
    while (!plane->empty()) {
        
        auto nextPlane = new vector<pair<TreeNode *, int>>();
        for (auto &pair : *plane){
            
            //            printf("[%d, %d]",pair.first->val, pair.second);
            
            if (pair.first->left) {
                nextPlane->push_back({pair.first->left, pair.second-1});
                
                if (pair.second == min) {
                    result.insert(result.begin(), {pair.first->left->val});
                    min--;
                }else{
                    result[pair.second-min-1].push_back(pair.first->left->val);
                }
            }
            if (pair.first->right) {
                nextPlane->push_back({pair.first->right, pair.second+1});
                
                if (pair.second == max) {
                    result.push_back({pair.first->right->val});
                    max++;
                }else{
                    result[pair.second-min+1].push_back(pair.first->right->val);
                }
            }
        }
        
        //        printf("\n");
        
        free(plane);
        plane = nextPlane;
    }
    
    free(plane);
    return result;
}

inline string singleMultiply(string &num, int digit){
    if (digit==0) {
        return "0";
    }
    string result = num;
    int carry = 0;
    for (int i = (int)num.length()-1; i>= 0; i--) {
        int multi = (num[i]-'0')*digit+carry;
        result[i] = multi%10+'0';
        carry = multi/10;
    }
    
    if (carry) {
        result.insert(result.begin(), carry+'0');
    }
    
    return result;
}

//把后一个加到前一个上面
inline void plusStringNum(string &num1, string &num2){
    int i = (int)num1.length()-1, j = (int)num2.length()-1;
    
    int carry = 0;
    while (i >= 0 && j >= 0) {
        int plus = num1[i]+num2[j]-'0'+carry;
        if (plus > '9') {
            carry = 1;
            num1[i] = plus-10;
        }else{
            carry = 0;
            num1[i] = plus;
        }
        
        i--;
        j--;
    }
    
    if (i < 0) {
        num1.insert(num1.begin(), num2.begin(), num2.begin()+j+1);
        i = j;
    }
    
    while (carry > 0 && i >= 0) {
        num1[i] += carry;
        if (num1[i] > '9') {
            num1[i] -= 10;
            carry = 1;
        }else{
            carry = 0;
        }
        
        i--;
    }
    
    if (carry > 0) {
        num1.insert(num1.begin(), '1');
    }
}

//656. Multiply Strings
string multiply(string &num1, string &num2) {
    
    string &shorter = num1.length() > num2.length() ? num2:num1;
    string &longer = num1 == shorter ? num2:num1;
    
    string result = "0";
    for (int i = (int)shorter.size()-1; i>=0; i--) {
        auto oneMulti = singleMultiply(longer, shorter[i]-'0');
        if (oneMulti.front() != '0') {
            oneMulti += string(shorter.size()-i-1, '0');
            plusStringNum(result, oneMulti);
            //            cout<<oneMulti<<", "<<result<<endl;
        }
    }
    
    return result;
}

//只包含小写字母的字符串可以看成是一个26进制的数，这个函数完成乘法
inline string letterMultiPly(string &str, int digit){
    if (digit==0) {
        return "a";
    }
    string result = str;
    int carry = 0;
    for (int i = (int)str.length()-1; i>= 0; i--) {
        int multi = (str[i]-'a')*digit+carry;
        result[i] = multi%26+'a';
        carry = multi/26;
    }
    
    if (carry) {
        result.insert(result.begin(), carry+'a');
    }
    
    return result;
}

string encode(vector<string> &strs) {
    
    string result = "";
    for (auto &str : strs){
        result += str+",";
    }
    
    result.pop_back();
    
    return result;
}

vector<string> decode(string &str) {
    
    vector<string> result = {""};
    
    for (auto iter = str.begin(); iter != str.end(); iter++) {
        if (*iter == ',') {
            result.push_back("");
        }else{
            result.back().push_back(*iter);
        }
    }
    
    return result;
}

vector<int> countBits(int num) {
    
    vector<int> result(num+1, 0);
    
    int count = 1;
    auto cur = result.begin()+count;
    while (count <= (num+1)/2) {
        
        auto end = cur;
        for (auto iter = result.begin(); iter != end; iter++) {
            *cur++ = 1+*iter;
        }
        count <<= 1;
    }
    
    auto iter = result.begin();
    for (int i = 0; i<=num-count; i++) {
        *cur++ = 1+*iter++;
    }
    
    return result;
}

//666. Guess Number Higher or Lower II
//比较难转化的动态规划题，难从现实情境转变过来
int getMoneyAmount(int n) {
    int amounts[n][n+1];
    
    for (int i = 0; i<n; i++) {
        memset(amounts[i], 0, sizeof(amounts[i]));
    }
    
    for (int i = 1; i<n; i++) {
        for (int j = i-1; j>=0; j--) {
            
            int minA = INT_MAX;
            for (int k = j; k<=i; k++) {
                int cur = k+1+max(amounts[j][k-j], amounts[k+1][i-k]);
                minA = min(minA, cur);
                
                printf("    [%d, %d](%d)?[%d, %d](%d)+%d=%d\n",j+1,k,amounts[j][k-j],k+2,i+1,amounts[k+1][i-k],k+1,cur);
            }
            amounts[j][i-j+1] = minA;
        }
        
        printf("%d: %d\n",i+1,amounts[0][i+1]);
    }
    
    return amounts[0][n];
}

/*
 对比[6,16]和[1,11]: [1, 11]的解时7->9,选7之后左边的解是3->5,和为8小于9。如果[6,16]按照[1, 11]的方式，先选12,左边的解3->5会增加10，而右边的解扩张5，导致左边的解变大。所以同样长度的序列，在解的取法上是不同的，因为在对比左右的时候，两边的长度会不同，更长的解的扩张值会更大，导致原来小的一边变成了大的一遍。
 所以只是按照长度来计算还是不行。
 */
int getMoneyAmount2(int n) {
    int amounts[n+1];
    int soluCount[n+1];  //这个记录的是最优解的长度
    
    amounts[0] = amounts[1] = 0;
    soluCount[0] = soluCount[1] = 0;
    
    for (int i = 1; i<n; i++) {
        int minA = INT_MAX;
        int numCount = 1;
        for (int j = 0; j<=i; j++) {
            int right = amounts[i-j]+soluCount[i-j]*(j+1);
            int cur = j+1+max(amounts[j], right);
            if (cur < minA) {
                minA = cur;
                numCount = amounts[j] > right ? soluCount[j]+1:soluCount[i-j]+1;
            }
            
            printf("     %d+[%d, %d](%d) = %d\n", j+1,amounts[j] > right?1:j+2,amounts[j] > right?j:i+1,amounts[j] > right?amounts[j]:right,cur);
        }
        
        printf("%d: min %d count %d\n",i+1, minA, numCount);
        
        amounts[i+1] = minA;
        soluCount[i+1] = numCount;
    }
    
    return amounts[n];
}

//667. 最长的回文序列
//空间利用只有右上角半边
int longestPalindromeSubseq(string &s) {
    int len[s.size()][s.size()];
    
    for (int i = 0; i<s.size(); i++) {
        len[i][i] = 1;
        for (int j = i-1; j>=0; j--) {
            len[j][i] = max(len[j+1][i], len[j][i-1]);
            if (s[i] == s[j]) {
                len[j][i] = max(len[j][i], (j<i-1?len[j+1][i-1]:0)+2);
            }
        }
    }
    
    return len[0][s.size()-1];
}

inline void convertStrToNum(string &str, int &num, int &highestOne, int &oneCount){
    for (int i = 0; i<str.size(); i++){
        if (str[i] == '1') {
            num |= 1<<(str.size()-i-1);
            if (highestOne < 0) highestOne = (int)str.size()-i;
            oneCount++;
        }
    }
}

//668. 一和零
//理解错了，这个函数求的是满足条件里的最大的数了，而实际是求满足条件的最大个数
int findMaxForm2(vector<string> &strs, int m, int n) {
    if (strs.empty()) {
        return 0;
    }
    
    int result = 0, highestOne = 0;
    
    for (int i = 0; i<strs.size(); i++) {
        string &str = strs[i];
        
        int oneCount;
        if (result == 0) {
            convertStrToNum(str, result, highestOne, oneCount);
            if (oneCount > n || str.size()-oneCount > m) { //不满足，重置
                result = 0;
            }
        }else{
            if (str.size() < highestOne) {
                continue;
            }
            
            bool fail = true;
            int maxForeZero = min((int)str.size()-highestOne, m);
            for (int i = 0; i<maxForeZero; i++){
                if (str[i] == '1') {
                    fail = false;
                }
            }
            
            if (!fail) {
                int cur = 0, curHOne = 0;
                convertStrToNum(str, cur, curHOne, oneCount);
                if (oneCount <= n && str.size()-oneCount <= m && cur > result) { //不满足，重置
                    result = cur;
                    highestOne = curHOne;
                }
            }
        }
    }
    
    return result;
}

//678. Shortest Palindrome
string convertPalindrome(string &str) {
    int maxLen = 1;
    string prefix = "";
    for (int i = str.size()-1; i>=0; i--) {
        
        int l = 0, r = i;
        while (l<r && str[l] == str[r]) {
            l++;
            r--;
        }
        if (l >= r) {
            return prefix+str;
        }
        
        prefix.push_back(str[i]);
    }
    
    //不可能调用
    return str;
}

inline bool canMatchWord(string &s, int start,const string &word){
    if (s.size()-start<word.size() || word.empty()) {
        return false;
    }
    
    int i = start, j = 0;
    while (j<word.size() && s[i] == word[j]) {
        i++;
        j++;
    }
    
    return j==word.size();
}

inline void lowerCase(string &str){
    for (int i = 0; i<str.size(); i++) {
        if (str[i] < 92) {
            str[i] += 32;
        }
    }
}

//683 单词拆分
//一个标准的动态规划题；k的情况依托于0-k之间的某些的最优解，连接的原则就是开头匹配的单词的长度。不同的题目变化的就是这个连接原则。
int wordBreak3(string& s, unordered_set<string>& dict) {
    if (s.empty()) {
        return 0;
    }
    int count[s.size()+1];
    count[s.size()] = 1;
    
    lowerCase(s);
    
    unordered_set<string> lowerDict;
    for (auto &word : dict){
        string cp = word;
        lowerCase(cp);
        lowerDict.insert(cp);
    }
    
    for (int i = s.size()-1; i>=0; i--) {
        
        printf("\n %d: %c \n",i,s[i]);
        
        int cur = 0;
        for (auto &word : lowerDict){
            if (canMatchWord(s, i, word)) {
                cout<<word<<" "<<count[i+word.size()]<<endl;
                cur += count[i+word.size()];
            }
        }
        printf("结果 %d\n-----------------\n",cur);
        count[i] = cur;
    }
    
    return count[0];
}

//685. 数据流中第一个唯一的数字
int firstUniqueNumber(vector<int> &nums, int number) {
    map<int, int> sits;
    bool findEnd = false;
    for (int i = 0; i<nums.size(); i++){
        if (nums[i] == number) {
            findEnd = true;
            break;
        }
        if (sits.find(nums[i]) == sits.end()) {
            sits[nums[i]] = i;
        }else{
            sits[nums[i]] = (int)nums.size();
        }
    }
    
    if (!findEnd) {
        return -1;
    }
    
    int minNum = -1, minIndex = (int)nums.size();
    for (auto pair : sits){
        if (pair.second < minIndex) { //把重复的值设成了nums.size()这个不可能的值，把取最小值和排除重复的统一了
            minIndex = pair.second;
            minNum = pair.first;
        }
    }
    
    if (minNum < 0) {
        return number;
    }
    
    return minNum;
}

//689. Two Sum IV - Input is a BST
//使用排序数组求和的思路，只是在跳到下一个更大数或更小数的时候，需要根据二叉搜索树的性质来
//可惜没法跳跃到父节点，本来是可以很好解决的
vector<int> twoSum(TreeNode * root, int n) {
    return {};
}

//691. Recover Binary Search Tree
//这题意外凯快速，本以为转为数组会很费时费空间，但想到数组存的只是指针，一个指针才一个字节，总的也没多少消耗
TreeNode * bstSwappedNode(TreeNode * root) {
    vector<TreeNode *> list;
    TreeNode::inorderList(list, root);
    
    int first = -1, second = -1;
    for (int i = 1; i<list.size(); i++) {
        if (list[i]->val < list[i-1]->val) {
            if (first < 0) {
                first = i-1;
            }else if (second < 0){
                second = i;
                break;
            }
        }
    }
    
    if (first >= 0) {
        if (second < 0 ) { //相邻的两个调换了
            TreeNode::swap(list[first], list[first+1]);
        }else{
            //隔开的两个调换了，那么必定是左边错的那对里左边是调换的，右边那对里右边是错的。
            TreeNode::swap(list[first], list[second]);
        }
    }
    
    return root;
}

int slidingWindowUniqueElementsSum(vector<int> &nums, int k) {
    map<int, int> count;
    int result = 0;
    
    for (int i = 0; i<min((int)nums.size(), k); i++) {
        count[nums[i]]++;
    }
    
    for (auto &pair : count){
        if (pair.second == 1) {
            result++;
        }
    }
    
    int lastCount = result;
    for (int i = k; i<nums.size(); i++) {
        
        int curCount = lastCount;
        int right = ++count[nums[i]];
        if (right == 1) {
            curCount++;
        }else if (right == 2){
            curCount--;
        }
        int left = --count[nums[i-k]];
        if (left == 1) {
            curCount++;
        }else if (left == 0){
            curCount--;
        }
        
        printf("%d+%d=%d\n",result,curCount, result+curCount);
        result += curCount;
        lastCount = curCount;
    }
    return result;
}

//698. Maximum Distance in Arrays
int maxDiff(vector<vector<int>> &arrs) {
    int minNum1 = INT_MAX, minNum2 = INT_MAX, maxNum1 = INT_MIN, maxNum2 = INT_MIN;
    int index1= -1, index2 = -1, index3 = -1, index4 = -1;
    
    int i = 0;
    for (auto &arr : arrs){
        if (arr.front() < minNum1) {
            minNum2 = minNum1;
            index2 = index1;
            
            minNum1 = arr.front();
            index1 = i;
        }else if (arr.front() < minNum2){
            minNum2 = arr.front();
            index2 = i;
        }
        
        if (arr.back() > maxNum2) {
            maxNum1 = maxNum2;
            index3 = index4;
            
            maxNum2 = arr.back();
            index4 = i;
        }else if (arr.back() > maxNum1){
            maxNum1 = arr.back();
            index3 = i;
        }
        
        i++;
    }
    
    if (index1 != index4) {
        return maxNum2-minNum1;
    }else{
        return max(maxNum2-minNum2, maxNum1-minNum1);
    }
}

//数1，给一个数，求它的二进制形式里有多少个1
inline int countOne(int num){
    int count = 0;
    while (num>0) {
        count += num&1;
        num >>= 1;
    }
    return count;
}

//706. Binary Watch
vector<string> binaryTime(int num) {
    vector<vector<string>> hours = {
        {"0"},
        {"1", "2", "4", "8"},
        {"3", "5", "6", "9", "10"},
        {"7", "11"}
    };
    
    vector<vector<string>> minutes = {
        {"00"},
        {"01", "02", "04", "08", "16", "32"},
        {"03", "05", "06", "09", "10", "12", "17", "18", "20", "24", "33", "34", "36", "40", "48"},
        {"07", "11", "13", "14", "19", "21", "22", "25", "26", "28", "35", "37", "38", "41", "42", "44", "49", "50", "52", "56"},
        {"15", "23", "27", "29", "30", "39", "43", "45", "46", "51", "53", "54", "57", "58"},
        {"31", "47", "55", "59"}
    };
    
    vector<string> result;
    for (int i = max(0, num-5); i<=min(num, 3); i++) {
        for (auto &h : hours[i]) {
            for (auto &m : minutes[num-i]){
                result.push_back(h+":"+m);
            }
        }
    }
    
    sort(result.begin(), result.end());
    
    return result;
}



int repeatedString(string &A, string &B) {
    if (A.size() >= B.size()) {
        for (int k = 0; k<=A.size()-B.size(); k++) {
            int i = 0, j = 0;
            while (A[i+k] == B[j]) {
                i++;
                j++;
            }
            if (j == B.size()) {
                return 1;
            }
        }
    }
    
    for (int k = (int)min(A.size(), B.size()-1); k>0; k--) { //先占掉更多的
        
        bool fail = false;
        int start = (int)A.size()-k;
        for (int i = 0; i<k; i++) {
            if (A[start+i] != B[i]) {
                fail = true;
                break;
            }
        }
        
        if (fail) {
            continue;
        }
        
        int BI = k, AI = 0;
        while (BI < B.size()) {
            if (A[AI] != B[BI]) {
                fail = true;
                break;
            }
            
            AI++;
            if (AI==A.size()) {
                AI = 0;
            }
            BI++;
        }
        
        if (!fail) {
            return ceilf((B.size()-k)/(float)A.size())+1;
        }
    }
    
    return -1;
}

//741. Calculate Maximum Value II  719. 计算最大值跟这个类似，但是计算顺序一定是从前往后的
//复杂度应该是O(n^3)
/*
 一个大问题看成一个小问题的边界，如整题的解等于求[i,n)的最大值在i=0时的特殊情况。
 而跟汉诺塔问题类似，这条思路可能会执行多次，在这里就是3次，每次降级，分别是:
 1. [0,n)
 2. 求[i,n),i从n-1到0
 3. 求[i，j],j从i到n-1
 4. 求[i, k],k从i到j。
 
 每个上级的解又是多次下级的解合并的起来的，这中分级嵌套的思想跟九连环类似。
 */
int calcMaxValue(string &str) {
    int maxVal[str.size()][str.size()];
    
    int nums[str.size()];
    for (int i = 0; i<str.size(); i++){
        nums[i] = str[i]-'0';
    }
    
    for (int i = (int)str.size()-1; i>=0; i--) {
        maxVal[i][i] = str[i] - '0';
        for (int j = i+1; j<str.size(); j++) {
            
            int curMax = INT_MIN;
            for (int k = i; k<j; k++) {
                int left = maxVal[i][k], right = maxVal[k+1][j];
                int temp = (left > 1 && right > 1)?(left*right):(left+right);
                if (temp > curMax) {
                    curMax = temp;
                    //                    printf("      [%d, %d] %d: %d\n",i,j,k,temp);
                }
            }
            
            maxVal[i][j] = curMax;
            
            //            printf("[%d, %d] %d\n",i,j,curMax);
        }
    }
    
    return maxVal[0][str.size()-1];
}

int leftRotate(int n, int d) {
    int bit = 32;
    return (n<<d)|(n>>(bit-d));
}

int computeLastDigit(long long A, long long B) {
    if (B-A >= 5) {
        return 0;
    }else if (B-A == 1){
        return B%10;
    }
    
    int d1 = A%10+1, d2 = B%10;
    if (d2 < d1) {
        d2 += 10;
    }
    
    int result = 1;
    for (int i = d1; i<=d2; i++) {
        result *= i;
    }
    
    return result%10;
}

void arrayReplaceWithGreatestFromRight(vector<int> &nums) {
    if (nums.empty()) {
        return;
    }
    
    int curMax = nums.back();
    nums[nums.size()-1] = -1;
    
    for (int i = nums.size()-2; i>=0; i--) {
        int cur = nums[i];
        nums[i] = curMax;
        curMax = max(cur, curMax);
    }
}

int monotoneDigits(int num) {
    int nc = num, index = 9;
    short digits[10];
    while (nc > 0) {
        digits[index] = nc%10;
        nc /= 10;
        index--;
    }
    index++;
    
    //1. 从前往后找到第一个减小的位置(AB) 2. 然后从A开始往回找，找到第一个不相等的位置或到开头，如PAAAAAB,P就是需要的地方，然后变成P(A-1)99999...，即从第一个A位置开始降位
    //实际代码修改：i和j两个位置，两者保持前后的位置，相等的时候，j向下移动，i不移动。这样在完成上面第一步时，就不需要执行第二步了，i就是需要的位置。
    //把i之后的数求出来，用num减掉，再减去1，就可以达到P(A-1)99999...的效果
    int i = index, j = index+1;
    while (j<10) {
        if (digits[j] > digits[i]) {
            i = j;
            j++;
        }else if (digits[j] == digits[i]){
            j++;
        }else{
            int remove = 0, weight = 1;
            for (int k = 9; k>i; k--) {
                remove += weight*digits[k];
                weight *= 10;
            }
            
            return num-remove-1;
        }
    }
    
    return num;
}

int minElements(vector<int> &arr) {
    sort(arr.begin(), arr.end());
    
    int sum = 0;
    for (auto &num : arr){
        sum += num;
    }
    sum /= 2;
    sum += 1;
    
    int count = 0, halfSum = 0;
    auto iter = arr.end()-1;
    while (halfSum < sum) {
        halfSum += *iter;
        count++;
    }
    
    return count;
}

vector<vector<string>> groupAnagrams(vector<string> &strs) {
    int counts[strs.size()][26];
    
    for (int i = 0; i < strs.size(); i++){
        auto &co = counts[i];
        memset(co, 0, sizeof(co));
        
        auto &str = strs[i];
        for (auto &c : str){
            co[c-'a']++;
        }
    }
    
    vector<vector<string>> result;
    int frontIndex[strs.size()];
    
    int i = 0;
    for (auto &str : strs){
        
        bool find = false;
        for (int k = 0; k<result.size(); k++){
            bool same = true;
            
            for (int j = 0; j<26; j++) {
                if (counts[i][j] != counts[frontIndex[k]][j]) {
                    same = false;
                    break;
                }
            }
            if (same) {
                result[k].push_back(str);
                find = true;
                break;
            }
        }
        
        if (!find) {
            frontIndex[result.size()] = i;
            result.push_back({str});
        }
        
        i++;
    }
    
    return result;
}

vector<string> findStrobogrammatic(int n) {
    
    if (n<2){
        if (n == 1) {
            return {"0","1","8"};
        }else if(n==0){
            return {""};
        }
    }
    
    int candCount = 5;
    char cands1[] = {'0','1','8','6','9'};
    char cands2[] = {'0','1','8','9','6'};
    
    vector<string> result;
    for (int i = 0; i<candCount-1; i++) { //开头不要0
        result.push_back(string(n,' '));
        result[i][0] = cands1[i+1];
        result[i][n-1] = cands2[i+1];
    }
    for (int i = 1; i<n/2; i++) {
        
        int size = (int)result.size();
        for (int j = 1; j<candCount; j++) {  //复制多份
            int orId = 0;
            while (orId < size) {
                string &cp = result[orId];
                cp[i] = cands1[j];
                cp[n-i-1] = cands2[j];
                result.push_back(cp);
                orId++;
            }
        }
        
        int orId = 0;
        while (orId < size) {
            string &cp = result[orId];
            cp[i] = cands1[0];
            cp[n-i-1] = cands2[0];
            orId++;
        }
    }
    
    if (n & 1) { //奇数,中间可加入0 1 8，复制两份加入1和8，原有那份填入0
        int size = (int)result.size(), mid = n/2;
        for (int i = 1; i<3; i++) {
            int orId = 0;
            while (orId < size) {
                string &cp = result[orId];
                cp[mid] = cands1[i];
                result.push_back(cp);
                orId++;
            }
        }
        
        int orId = 0;
        while (orId < size) {
            string &cp = result[orId];
            cp[mid] = '0';
            orId++;
        }
    }
    
    return result;
}

enum MazeDirection{
    None,
    Up,
    Down,
    Left,
    Right,
};

struct MazeState{
    int x;
    int y;
    MazeDirection dir;
    
    MazeState(MazeDirection dir, int x, int y):dir(dir),x(x),y(y){};
};

inline bool checkPoint(vector<vector<int>> &maze, int x, int y){
    int width = (int)maze.size(), height = (int)maze.front().size();
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return false;
    }
    if (maze[x][y] == 1) {
        return false;
    }
    
    return true;
}

bool checkDest(vector<vector<int>> &maze, int x, int y){
    bool left = checkPoint(maze, x-1, y);
    bool right = checkPoint(maze, x+1, y);
    bool up = checkPoint(maze, x, y-1);
    bool down = checkPoint(maze, x, y+1);
    
    if ((!left && !right && up && down) || (left && right && !up && !down)) {
        return false;
    }
    return true;
}

bool hasPath(vector<vector<int>> &maze, vector<int> &start, vector<int> &destination) {
    
    if (!checkDest(maze, destination.front(), destination.back())) {
        return false;
    }
    
    int width = (int)maze.size(), height = (int)maze.front().size();
    
    //只记录拐弯点
    bool visited[width][height];
    for (int i = 0; i<maze.size(); i++) {
        memset(visited[i], 0, sizeof(visited[i]));
    }
    
    MazeState curState(None, start.front(), start.back());
    Point dest = Point(destination.front(), destination.back());
    stack<MazeState> forks;
    
    do {
        
        if (curState.x == dest.x && curState.y == dest.y) {
            return true;
        }
        
        //第一步： 找一个新的起点
        MazeState next(None, -1, -1);
        
        //这里可以根据出口位置来确定优先走的方向
        if (!visited[curState.x][curState.y]) { //当前拐点已经判断过了，直接回溯
            if (curState.dir != Left && curState.dir != Right) {
                if (checkPoint(maze, curState.x+1, curState.y)) {
                    next = MazeState(Right, curState.x+1, curState.y);
                }
                if (checkPoint(maze, curState.x-1, curState.y)) {
                    
                    if (next.x > 0) {
                        forks.push(MazeState(Left, curState.x-1, curState.y));
                    }else{
                        next = MazeState(Left, curState.x-1, curState.y);
                    }
                }
            }
            
            if (curState.dir != Up && curState.dir != Down) {
                if (checkPoint(maze, curState.x, curState.y+1)) {
                    if (next.x > 0) {
                        forks.push(MazeState(Down, curState.x, curState.y+1));
                    }else{
                        next = MazeState(Down, curState.x, curState.y+1);
                    }
                }
                if (checkPoint(maze, curState.x, curState.y-1)) {
                    if (next.x > 0) {
                        forks.push(MazeState(Up, curState.x, curState.y-1));
                    }else{
                        next = MazeState(Up, curState.x, curState.y-1);
                    }
                }
            }
        }
        
        //当前点的四周没有，就回溯到上一个岔路口，如果没有路口了，结束失败
        if (next.x < 0) {
            if (forks.empty()) {
                return false;
            }
            curState = forks.top();
            forks.pop();
        }else{
            visited[curState.x][curState.y] = true;  //新的路口都要标记
            curState = next;
        }
        
        printf("\n-------------\n");
        ///从新的起点，按照既定方向运动，直到碰到边界或墙
        int nextX = curState.x, nextY = curState.y;
        do {
            
            printf("(%d, %d) ",nextX, nextY);
            
            if (curState.dir == Left) {
                nextX--;
            }else if (curState.dir == Right){
                nextX++;
            }else if (curState.dir == Up){
                nextY--;
            }else{
                nextY++;
            }
            
            if (!checkPoint(maze, nextX, nextY)) {
                break;
            }
            
            curState.x = nextX;
            curState.y = nextY;
            
        } while (1);
        
    } while (1);
    
    return false;
}

static const int CornerNodeLinkSize = 4;
//路径上每个拐点
struct CornerNode{
    int x;
    int y;
    int dis;  //离起点距离
    int endDis = -1;
    CornerNode *linkedNodes[CornerNodeLinkSize] = {0};
    
    int mark = 0;  //0 代表还没查找它的连接点，1-4代表访问到第几个岔路了，上下左右对应1-4，其他的都代表访问结束。5代表死路
    
    CornerNode(int x, int y, int dis):x(x),y(y),dis(dis){};
};

//给定方向，找到下一个联接拐点
inline CornerNode *findLinkedNode(vector<vector<int>> &maze, CornerNode *node, MazeDirection dir, CornerNode **genNodes){
    
    CornerNode *nextNode = nullptr;
    int nextX = node->x, nextY = node->y;
    
    do {
        if (dir == Left) {
            nextX--;
        }else if (dir == Right){
            nextX++;
        }else if (dir == Up){
            nextY--;
        }else{
            nextY++;
        }
        
    } while (checkPoint(maze, nextX, nextY));
    
    int dis = 0;
    if (dir == Left) {
        nextX++;
        dis = node->x-nextX;
    }else if (dir == Right){
        nextX--;
        dis = nextX-node->x;
    }else if (dir == Up){
        nextY++;
        dis = node->y-nextY;
    }else{
        nextY--;
        dis = nextY-node->y;
    }
    
    if (dis > 0) {
        auto index = nextY*maze.size()+nextX;
        nextNode = genNodes[index];
        if (nextNode == nullptr) {
            nextNode = new CornerNode(nextX, nextY, node->dis+dis);
            genNodes[index] = nextNode;
            //            printf("new %d (%d,%d)\n",index,nextX,nextY);
        }else{
            nextNode->dis = min(nextNode->dis, node->dis+dis); //可能后面找到更短距离
        }
    }
    
    return nextNode;
}

inline int markEndDis(vector<CornerNode *> &path, CornerNode *lastNode){
    
    printf("\n\n*******over******* (%d, %d)%d %d\n",lastNode->x, lastNode->y,lastNode->dis, lastNode->endDis);
    
    int fullLen = lastNode->dis+lastNode->endDis;
    for (auto &node : path){
        if (node->endDis>=0) {
            node->endDis = min(node->endDis, fullLen-node->dis);
        }else{
            node->endDis = fullLen-node->dis;
        }
        
        printf("(%d,%d)%d ",node->x, node->y, node->endDis);
    }
    
    printf("\nfull: %d \n\n",fullLen);
    
    return fullLen;
}

//可以把每个拐点做成一个节点，然后整体构成一个有向图，这样会清晰一些，每个节点保持上下左右的联通，这样可以抛掉那些无用的点
//1. 记录沿途所有拐点，并标记为已访问 2.找到出口或者死路或者环，就退回上一个有新分叉口的拐点，开始新的路。3. 只有找到出口回退的时候，才
//按照上面3点就可以把多有路径遍历完，可以把建图和遍历一起完成。找到一个新的点，把上下左右的可能拐点全部找到，，关联到当前点。

//还可以优化：当前点来的方向的反方向不用去检测了，因为之前的节点一定会探索到，这里掉头反而是重复了. **但是提交代码竟然慢了，奇怪**
int shortestDistance(vector<vector<int>> &maze, vector<int> &start, vector<int> &destination) {
    if (!checkDest(maze, destination.front(), destination.back())) {
        return -1;
    }
    
    int width = (int)maze.size(), height = (int)maze.front().size();
    
    CornerNode *curNode = new CornerNode(start.front(), start.back(), 0);
    int destX = destination.front(), destY = destination.back();
    vector<CornerNode *> path;
    int minPath = INT_MAX;
    
    //已经生成的节点
    //    CornerNode **genNodes = new CornerNode*[width*height];
    //    memset(genNodes, 0, width*height*sizeof(CornerNode*));
    CornerNode *genNodes[width*height];
    memset(genNodes, 0, sizeof(genNodes));
    genNodes[width*curNode->y+curNode->x] = curNode;
    
    while (1) {
        
        //在这里curNode有2种可能：0 新点(包含终点) 1-4 回溯后的点
        printf("(%d, %d) ",curNode->x, curNode->y);
        
        //找到终点
        if (curNode->x == destX && curNode->y == destY) {
            curNode->endDis = 0;
            curNode->mark = CornerNodeLinkSize+1;
            
            int fullLen = markEndDis(path, curNode);
            minPath = min(minPath, fullLen);
        }
        
        //1. 还未探索，构建连接节点
        if (curNode->mark == 0) {
            for (int i = 0; i<CornerNodeLinkSize; i++) {
                curNode->linkedNodes[i] = findLinkedNode(maze, curNode, (MazeDirection)(i+1), genNodes);
            }
            
            curNode->mark = 1;
        }
        
        //2. 先查找当前节点的连接点，找到新的点
        CornerNode *nextNode = nullptr;
        if (curNode->mark <= CornerNodeLinkSize) {
            
            while (curNode->mark <= CornerNodeLinkSize) {
                nextNode = curNode->linkedNodes[curNode->mark-1];
                curNode->mark++; //比当前的大1，最后会停在5（CornerNodeLinkSize+1）
                
                if (nextNode) {
                    if (nextNode->mark == 0) {
                        path.push_back(curNode);
                        break;
                    }else if (nextNode->mark > CornerNodeLinkSize){  //死路，结束了的
                        
                        if (nextNode->endDis >= 0) { //去过终点的点
                            int fullLen = markEndDis(path, nextNode);
                            minPath = min(minPath, fullLen);
                        }
                        nextNode = nullptr;  //否则调到这个点去了
                    }
                    //1-4 环
                }
            }
        }
        
        //3. 在当前节点没有找到连接点，回退到一个有岔路的可能点
        if (nextNode == nullptr) {
            printf("\n");
            while (!path.empty()) {
                nextNode = path.back();
                path.pop_back();
                
                if (nextNode->mark <= CornerNodeLinkSize) {
                    break;
                }
            }
            if (nextNode == nullptr) {
                return minPath == INT_MAX?-1:minPath;
            }
        }
        
        curNode = nextNode;
    }
}

int intersectionOfArrays(vector<vector<int>> &arrs) {
    int minRight = INT_MAX, maxLeft = INT_MIN;
    int arrCount = arrs.size();
    
    for (auto &arr : arrs){
        if (arr.empty()) {
            return 0;
        }
        int left = INT_MAX, right = INT_MIN;
        for (auto &num : arr){
            left = min(left, num);
            right = max(right, num);
        }
        minRight = min(minRight, right);
        maxLeft = max(maxLeft, left);
    }
    
    if (minRight < maxLeft) {
        return 0;
    }
    
    int size = minRight-maxLeft+1;
    short count[size];
    memset(count, 0, sizeof(count));
    
    for (auto &arr : arrs){
        for (auto &num : arr){
            if (num>=maxLeft && num<=minRight) {
                count[num-maxLeft]++;
            }
        }
    }
    
    int result = 0;
    for (int i = 0; i<size; i++) {
        if (count[i] == arrCount) {
            result++;
        }
    }
    
    return result;
}

static short reorderedAlphabet[26];
bool wordSortComp(const string &str1, const string str2){
    auto iter1 = str1.begin(), iter2 = str2.begin();
    while (iter1 != str1.end() && iter2 != str2.end()) {
        if (*iter1 == *iter2) {
            iter1++;
            iter2++;
        }else{
            return reorderedAlphabet[*iter1-'a'] < reorderedAlphabet[*iter2-'a'];
        }
    }
    
    return str1.size() <= str2.size();
}

vector<string> wordSort(string &alphabet, vector<string> &words) {
    int order = 0;
    for (auto &c : alphabet){
        reorderedAlphabet[c-'a'] = order++;
    }
    
    sort(words.begin(), words.end(), wordSortComp);
    return words;
}

//二分法找到开始时间和结束时间落在的区域，inside为true代表落在上线区间内，start表示查询的开始索引，因为后面的时间更晚
//inline int findPosition(vector<Interval> &seqA, bool *inside, int start){
//
//}
//
//vector<Interval> timeIntersection(vector<Interval> &seqA, vector<Interval> &seqB) {
//
//}

int getSingleNumber(vector<int> &nums) {
    if (nums.size() == 1 || nums[0] != nums[1]) {
        return nums.front();
    }
    
    int left = 0, right = (int)nums.size()-1;
    while (left < right-1) {
        int mid = left+(right-left)/2 | 1;
        if (nums[mid] == nums[mid-1]) {
            left = mid;
        }else{
            right = mid-1;
        }
    }
    
    return nums[right];
}

int threeSum2(int n) {
    int result = 0;
    
    int sqrtNum = sqrtf(n);
    for (int i = 0; i<=sqrtNum; i++) {
        int left = n-i*i;
        int sqrtNum2 = sqrtf(left/2);
        int sqetNum3 = sqrtf(left);
        for (int j = i; j<=sqrtNum2; j++) {
            int last = left-j*j;
            int k = sqrtf(last);
            if (k*k==last) {
                result++;
                
                printf("%d %d %d\n",i,j,k);
            }
        }
    }
    
    return result;
}

int countNumber2(vector<vector<int>> &nums, int left, int top, int right, int bottom){
    int count = 0;
    for (int i = left; i<=right; i++) {
        auto &line = nums[i];
        for (int j = top; j<=bottom; j++) {
            int val = line[j];
            if (line[j]<0) {
                count++;
            }
        }
    }
    
    return count;
}

int countNumber3(vector<vector<int>> &nums) {
    if (nums.empty()) {
        return 0;
    }
    
    int right = 0;
    auto &line = nums[0];
    if (line[0]>=0) {
        return 0;
    }else if (line.back() < 0){
        right = line.size()-1;
    }else{
        int i = 0, j = (int)line.size()-1;
        
        while (i<j-1) {
            int mid = i+(j-i)/2;
            if (line[mid]<0) {
                i = mid;
            }else{
                j = mid;
            }
        }
        right = i;
    }
    
    int result = right+1;
    for (int i = 1; i<nums.size(); i++) {
        auto &line = nums[i];
        while (right>=0 && line[right]>=0) {
            right--;
        }
        
        if (right<0) {
            break;
        }
        result += right+1;
    }
    
    return result;
}

/*
 1.当次的字符，镜像翻转后，1和0对调，加入到后面，如1011 --> 1011+1011的镜像的对调 -->1011+1101的对调 --> 1011+0010 --> 10110010。经过这个操作变为这次新增的字符串
 2.然后把当次新增的字符间隔的和上次的结果混合，***+#### --> #*#*#*# ,*表示上次的结果集，#表示这次产生的新字符串
 3. 有了上面的操作过程，得到推论：
 1) 有两种字符串，一个是当次新增，个数变化为1 2 4 8 ...2^(k-1);另一种是结果集，个数变化为1 3 7 15 ...2^k-1,每次个数增加就是新增字符的个数
 2) 经过k次折叠后，总数为2^k-1,这一次产生2^(k-1)个，这次里索引为d的折痕，再经过t次折叠后，索引为index,则：
 index = (2d+1)*(2^t-1)+2d, 2d是这一次插入到结果集后的索引，包括自己在内的2d+1个字符，每个前面都新增2^t-1，所以折叠了n次后，索引变为:
 index = 2^(n-k)*(2*d+1)-1, d的范围为[0, 2^(k-1)-1]。
 3) 那么对于经过k次折叠后的索引为d的折痕，它的值等于什么？
 d >= 2^(k-2),等于上一次的2^(k-1)-d-1的值的反值； d < 2^(k-2),就等于上次d的值
 */

#define GSEndIndex(k, d, n) ((1<<(n-(k)))*(2*(d)+1)-1)

string getString(int n) {
    
    int total = (1<<n)-1;
    bool *mem = (bool*)malloc(sizeof(bool)*total);
    memset(mem, 0, total);
    
    int step = total+1, frontC = step>>1;
    for (int i = 1; i<n; i++) {
        
        step >>= 1;
        frontC >>= 1;
        
        for (int j = frontC-1; j<total/2; j+=step) {
            mem[j] = mem[2*j+1];
            mem[total-j-1] = !mem[j];
            
            //            printf("%d-%d\n",j,total-j-1);
        }
    }
    
    string result(total, '0');
    int i = 0;
    for (auto &c : result){
        c += mem[i];
        //        printf("%d ",mem[i]);
        i++;
    }
    //    printf("\n");
    
    return result;
}

//843. 数字翻转
//做了这么多动态规划的题，这题还是让我惊讶，因为用了动态规划的思路去做后，变得如此简单，逻辑这么简洁！而常人的逻辑去思考，这个问题还挺复杂的
//动态规划的厉害之处在于：直接在最优解之间建立了联系，直接从k情况的最优解推到k+1情况的最优解，而不用每次都面对k种情况
int flipDigit(vector<int> &nums) {
    
    int opera = 0;
    int oneCount = 0;
    
    for (auto iter = nums.rbegin(); iter != nums.rend(); iter++) {
        if (*iter==1) {
            oneCount++;
        }else{
            opera = min(oneCount, opera+1);
        }
    }
    
    return opera;
}

//从16开始，就是从k-5复制4次,增长了维持在1.25-1.33
int maxA(int N) {
    vector<int> dp(N + 1, 0);
    for (int i = 0; i <= N; ++i) {
        dp[i] = i;
        int maxj = 1;
        for (int j = 1; j < i - 2; ++j) {
            int cur = dp[j] * (i - j - 1);
            if (cur>dp[i]) {
                dp[i] = cur;
                maxj = j;
            }
        }
        
        printf("%d->%d[%d] %d rate:%.3f\n",i,maxj,i-maxj,dp[i],dp[i]/(float)dp[i-1]);
    }
    return dp[N];
}

//869. 找出一个数组的错乱
//1: 当4换到1的位置，1不在4的位置时，这时的错乱情况可以看成原始状态为231的序列错了乱，因为231分别都不能在这样的位置上，而它的排列情况等价于123
//2. 缺乏对大数乘法求余的算法，这题解不出来
int findDerangement(int n) {
    if (n<3) {
        if (n==1) {
            return 0;
        }else if (n==2){
            return 1;
        }
    }
    int limit = 1000000007;
    int res1 = 0, res2 = 1;
    for (int i = 3; i<=n; i++) {
        int base = (res1+res2)%limit;
        
        int res = base;
        int rate = limit/base;
        if (rate<i-1) {
            int j = 0;
            for (; j<i-1; j+=rate) {
                res += base*rate;
                res %= limit;
            }
            res += base*(i-j-rate);
            res %= limit;
            
        }else{
            res = base*(i-1);
        }
        
        res1 = res2;
        res2 = res;
        
        printf("%d %d\n",i,res2);
    }
    
    return res2;
}

//871. 最小分解
int smallestFactorization(int a) {
    int factors[8];
    memset(factors, 0, sizeof(factors));
    
    for (int i = 9; i>1; i--) {
        while (a%i==0) {
            factors[i-2]++;
            a /= i;
        }
        if (a == 1) {
            break;
        }
    }
    
    if (a != 1) {
        return 0;
    }
    
    int result = 0;
    int bit = 1;
    int overCount = 0, overBit = 1;
    for (int i = 9; i>1; i--) {
        int j = factors[i-2];
        while (j>0) {
            if (bit == 1000000000) { //当位权值已经到了int最大位的时候，从新开始统计，即分成两部分：小于1000000000的值大小和大于这个值的大小。
                overCount += i*overBit;
                if (overCount > 2) { //int最大值最高位是2
                    return 0;
                }
                overBit *= 10;
                
            }else{
                result += i*bit;
                bit *= 10;
            }
            
            j--;
        }
    }
    
    if (overCount == 2) {
        if (INT_MAX-2000000000 < result) {
            return 0;
        }
    }
    
    return result;
}

inline int killProcessFind(vector<int> &ppid, int target){
    int cur = (int)ppid.size()-1;
    while (ppid[cur] != target) {
        cur--;
    }
    
    return cur;
}

//杀死进程，题目是要返回列表，这里的解法是求最后一个被杀死的
int killProcess(vector<int> &pid, vector<int> &ppid, int kill) {
    
    int target = kill;
    while (1) {
        int i = killProcessFind(ppid, target);
        if (i < 0) {
            return target;
        }
        target = pid[i];
    }
}

//873. 模拟松鼠
//看起来难，其实很简单，每次拿一个，每个坚果的任务是独立的，除了第一次，唯一的优化点就是第一次拿哪一个坚果。
int minDistance(int height, int width, vector<int> &tree, vector<int> &squirrel, vector<vector<int>> &nuts) {
    int result = 0;
    int treeX = tree.front(), treeY = tree.back();
    int squX = squirrel.front(), squY = squirrel.back();
    
    int minSquDiff = INT_MAX;
    for (auto &nut:nuts){
        int treeLen = abs(treeX-nut.front())+abs(treeY-nut.back());
        int squLen = abs(squX-nut.front())+abs(squY-nut.back());
        
        result += 2*treeLen;
        minSquDiff = min(minSquDiff, squLen-treeLen);
    }
    
    return result+minSquDiff;
}

//所有其他点和本点的连线，都在相邻连线构成的小于180度的那一边。
//或者从左边连线每个点一直到右边，角度变化的方向是不变的
bool isConvex(vector<vector<int>> &point) {
    float slope[point.size()][point.size()];
    for (int i = 0; i<point.size(); i++) {
        memset(slope[i], 0, sizeof(slope[i]));
    }
    
    for (int i = 0; i<point.size();i++){
        auto &p1 = point[i];
        for (int j = i+1;j<point.size();j++){
            auto &p2 = point[j];
            
            float diffX = p2.front()-p1.front();
            float diffY = p2.back()-p1.back();
            if (diffX == 0) {
                slope[i][j] = INT_MAX;
                slope[j][i] = INT_MAX;
            }else{
                slope[j][i] = slope[i][j] = diffY/diffX;
            }
            
            printf("%d->%d %.3f\n",i,j,slope[i][j]);
        }
    }
    
    for (int i = 0; i<point.size(); i++) {
        
        int pre = i==0?point.size()-1:i-1;
        float maxSlope, minSlope;
        if (slope[i][i+1]<slope[i][pre]) {
            minSlope = slope[i][i+1];
            maxSlope = slope[i][pre];
        }else{
            maxSlope = slope[i][i+1];
            minSlope = slope[i][pre];
        }
        for (int j = 0; j<point.size(); j++) {
            float val = slope[i][j];
            if (i!=j && (slope[i][j] < minSlope || slope[i][j] > maxSlope)) {
                return false;
            }
        }
    }
    
    return true;
}

void permuteUnique(vector<vector<int>> &res, vector<int> &nums, vector<int> &curList, bool *picked){
    
    int last = nums.back()+1;
    int index = 0;
    
    for (auto &num:nums){
        if (!picked[index] && num != last) {
            picked[index] = true;
            curList.push_back(num);
            if (curList.size() == nums.size()) {
                res.push_back(curList);
                curList.pop_back();
                picked[index] = false;
                return;
            }
            permuteUnique(res, nums, curList, picked);
            
            last = num;
            curList.pop_back();
            picked[index] = false;
        }
        index++;
    }
}

//16. 带重复元素的排列
//在当前可选的里面选一个作为下一个，然后推到下一个循环，这样就不会重复。因为在当前位置上，某个种类的数只会出现一次，在当前位置上不会重复。
vector<vector<int>> permuteUnique2(vector<int> &nums) {
    if (nums.empty()) {
        return {{}};
    }
    sort(nums.begin(), nums.end());
    bool picked[nums.size()];
    memset(picked, 0, sizeof(picked));
    
    vector<vector<int>> result;
    vector<int> curList;
    permuteUnique(result, nums, curList, picked);
    
    return result;
}

//16. 带重复元素的排列
//非递归版本，这就是“深度搜索”标签的原因
/*
 1. 深度搜索关键在于回溯，而回溯就需要记住每个选择点，即岔路，的状态，该定义一个什么样的状态又是其中关键问题。
 这题里状态就是：当前的路径、已选的数标记和当前选到第几个数
 2. 另一个关键点是：向前推进的行为和回溯后的行为要一致，这样才能统一成一个行为，即开始一个新的循环时，你不能区分它是新点还是回溯的旧点。一般就是把新店置为特殊状态的旧点，比如这题,新点就是挖掘0个子节点的点，而旧点就是>0的点，但它们的行为是一致的，就是从未选的数里选一个作为子节点。
 3. 第三个关键点就是边界条件：1）什么时候得到完整的路径，即得到一个解 2）什么时候回溯。第一个是在路径最深的时候，指标可以是深度或者没有新的路径了；第二个是当前没有新的路径了，所以**当前选择到哪条路了**是必须的状态
 */
vector<vector<int>> permuteUnique(vector<int> &nums) {
    if (nums.empty()) {
        return {{}};
    }
    sort(nums.begin(), nums.end());
    bool picked[nums.size()];
    memset(picked, 0, sizeof(picked));
    
    vector<vector<int>> result;
    vector<int> curList;
    stack<int> forks;  //这里的索引不是当前节点自己的索引，而是它当前有k个数可以选，从前往后的尝试，下一个可以尝试的数的索引。就是岔路口试到哪一条路了
    forks.push(0);
    
    while (1) {
        
        int &index = forks.top();
        int last = index==0?(nums.back()+1):nums[index-1];
        bool foundNext = false;
        
        while (index < nums.size()){
            auto num = nums[index];
            if (!picked[index] && num != last) {
                foundNext = true;
                
                curList.push_back(num);
                if (curList.size() == nums.size()) {
                    result.push_back(curList);
                    
                    curList.pop_back();
                    foundNext = false; //虽然找到了，但是只有唯一解，这题的特殊情况，所以回退，用这个变量造个假
                }else{
                    picked[index] = true;
                    index++;
                    forks.push(0);
                }
                
                break;
            }else{
                index++;
            }
        }
        
        if (!foundNext) {
            curList.pop_back();
            forks.pop();
            if (forks.empty()) {
                return result;
            }
            picked[forks.top()-1] = false;
        }
    }
    
    return result;
}

struct Compute24Node {
    int size;
    int first;
    int second;
    int ope = -1;
    
    Compute24Node(int size, int first, int second):size(size),first(first),second(second){};
};

inline double operateTwoNum(double num1, double num2, int op){
    switch (op) {
        case 0:
            return num1+num2;
            break;
        case 1:
            return num1-num2;
            break;
        case 2:
            return num2-num1;
            break;
        case 3:
            return num1*num2;
            break;
        case 4:
            return num1/num2;
            break;
        case 5:
            return num2/num1;
            break;
            
        default:
            break;
    }
    return 0;
}

static int compute24Time = 1;
//24 Game ，这题用深度搜索的角度写挺有意思
bool compute24(vector<double> &nums) {
    
    vector<double> nums3(3,0), nums2(2,0);
    stack<Compute24Node> path;
    path.push(Compute24Node(4, 0, 1));
    
    while (1) {
        
        Compute24Node &curNode = path.top();
        auto &curNums = (curNode.size == 4?nums:(curNode.size==3?nums3:nums2));
        
        if (curNode.ope < 5) { //加减乘除4种，但减和除有前后区别，再加两种
            curNode.ope++;
        }else{
            if (curNode.second < curNode.size-1) {
                curNode.second++;
                curNode.ope = 0;
            }else if(curNode.first < curNode.size-2){
                curNode.first++;
                curNode.second = curNode.first+1;
                curNode.ope = 0;
            }else{
                path.pop();
                if (path.empty()) {
                    break;
                }
                continue;
            }
        }
        
        auto operateNum = operateTwoNum(curNums[curNode.first], curNums[curNode.second], curNode.ope);
        if (curNode.size == 2) {
            if (abs(operateNum-24) < 1E5) {
                return true;
            }else{
                continue;
            }
        }
        
        auto &nextNums = curNode.size == 4 ? nums3:nums2;
        nextNums[0] = operateNum;
        
        int index = 1;
        for (int i = 0; i<curNums.size(); i++) {
            if (i != curNode.first && i!=curNode.second) {
                nextNums[index] = curNums[i];
                index++;
            }
        }
        
        path.push(Compute24Node(curNode.size-1, 0, 1));
        
    }
    
    return false;
}

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

#define enqueue(a,b) {string name = a;shelter.enqueue(name, b);}shelter.showAnimals();
#define dequeueAny() cout<<shelter.dequeueAny()<<endl;shelter.showAnimals();
#define dequeueDog() cout<<shelter.dequeueDog()<<endl;shelter.showAnimals();
#define dequeueCat() cout<<shelter.dequeueCat()<<endl;shelter.showAnimals();

class AnimalShelter {
    const int type_dog = 1;
    const int type_cat = 0;
    struct Animal{
        string name;
        int type;
        Animal(){
            name = "";
            type = -1;
        };
        Animal(string &name, int type):name(name),type(type){};
        
        Animal *pre = nullptr;
        Animal *next = nullptr;
        Animal *pre_sameType = nullptr;
        Animal *next_sameType = nullptr;
    };
    
    Animal *anyHead = new Animal();
    Animal *anyTail = new Animal();
    Animal *catHead = new Animal();
    Animal *catTail = new Animal();
    Animal *dogHead = new Animal();
    Animal *dogTail = new Animal();
    
    //node1->nodex ==> node1->node2->nodex
    inline void insert(Animal *node1, Animal *node2){
        node1->next->pre = node2;
        node2->next = node1->next;
        
        node1->next = node2;
        node2->pre = node1;
    }
    inline void insert_sameType(Animal *node1, Animal *node2){
        node1->next_sameType->pre_sameType = node2;
        node2->next_sameType = node1->next_sameType;
        
        node1->next_sameType = node2;
        node2->pre_sameType = node1;
    }
    
    //nodex->node1->node2 ==> nodex->node2
    inline void remove(Animal *node1, Animal *node2){
        node1->pre->next = node2;
        node2->pre = node1->pre;
        
        node1->pre = nullptr;
        node1->next = nullptr;
    }
    inline void remove_sameType(Animal *node1, Animal *node2){
        node1->pre_sameType->next_sameType = node2;
        node2->pre_sameType = node1->pre_sameType;
        
        node1->pre_sameType = nullptr;
        node1->next_sameType = nullptr;
    }
    
public:
    
    AnimalShelter(){
        anyTail->next = anyHead;
        anyHead->pre = anyTail;
        
        catTail->next_sameType = catHead;
        catHead->pre_sameType = catTail;
        
        dogTail->next_sameType = dogHead;
        dogHead->pre_sameType = dogTail;
    }
    
    void enqueue(string &name, int type) {
        auto newAni = new Animal(name, type);
        
        insert(anyTail, newAni);
        
        if (type == type_cat) {
            insert_sameType(catTail, newAni);
        }else{
            insert_sameType(dogTail, newAni);
        }
    }
    
    string dequeueAny() {
        if (anyHead==anyTail->next) {
            return "";
        }
        
        auto node = anyHead->pre;
        remove(anyHead->pre, anyHead);
        
        if (node->type == type_cat) {
            remove_sameType(catHead->pre_sameType, catHead);
        }else{
            remove_sameType(dogHead->pre_sameType, dogHead);
        }
        
        auto result = node->name;
        delete node;
        return result;
    }
    
    string dequeueDog() {
        if (dogHead==dogTail->next) {
            return "";
        }
        
        auto node = dogHead->pre_sameType;
        remove_sameType(dogHead->pre_sameType, dogHead);
        
        remove(node, node->next);
        
        auto result = node->name;
        delete node;
        return result;
    }
    
    string dequeueCat() {
        if (catHead==catTail->next) {
            return "";
        }
        
        auto node = catHead->pre_sameType;
        remove_sameType(catHead->pre_sameType, catHead);
        
        remove(node, node->next);
        
        auto result = node->name;
        delete node;
        return result;
    }
    
    void showAnimals(){
        printf("\n*******************\nany: ");
        auto cur = anyTail->next;
        while (cur != anyHead) {
            cout<<cur->name<<" ";
            cur = cur->next;
        }
        printf("\n");
        
        printf("cat: ");
        cur = catTail->next_sameType;
        while (cur != catHead) {
            cout<<cur->name<<" ";
            cur = cur->next_sameType;
        }
        printf("\n");
        
        printf("dog: ");
        cur = dogTail->next_sameType;
        while (cur != dogHead) {
            cout<<cur->name<<" ";
            cur = cur->next_sameType;
        }
        printf("\n");
    }
};

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
    int height = (int)matrix.size(), width = (int)matrix.front().size();
    if (k>height || k>width || k == 0) {
        return 0;
    }
    
    if (k==1) {
        int maxNum = INT_MIN;
        for (auto &row : matrix){
            for (auto &num : row){
                maxNum = max(maxNum, num);
            }
        }
        return maxNum;
    }
    
    int firstRow[width];
    memset(firstRow, 0, sizeof(firstRow));
    for (int i = 0; i<width; i++) {
        for (int j = 0; j<k; j++) {
            firstRow[i] += matrix[j][i];
        }
    }
    
    int firstColumn[height];
    memset(firstColumn, 0, sizeof(firstColumn));
    for (int i = 0; i<height; i++) {
        for (int j = 0; j<k; j++) {
            firstColumn[i] += matrix[i][j];
        }
    }
    
    int sum[height][width];
    for (int i = 0; i<height; i++) {
        memset(sum[i], 0, sizeof(sum[i]));
    }
    
    for (int i = 0; i<k; i++) {
        sum[k-1][k-1] += firstRow[i];
    }
    int maxSum = sum[k-1][k-1];
    
    for (int i = k; i<height; i++) {
        sum[i][k-1] = sum[i-1][k-1]+firstColumn[i]-firstColumn[i-k];
        maxSum = max(maxSum, sum[i][k-1]);
    }
    
    for (int i = k; i<width; i++) {
        sum[k-1][i] = sum[k-1][i-1]+firstRow[i]-firstRow[i-k];
        maxSum = max(maxSum, sum[k-1][i]);
    }
    
    
    for (int i = k; i<height; i++) {
        for (int j = k; j<width; j++) {
            sum[i][j] = sum[i-1][j]+sum[i][j-1]-sum[i-1][j-1]+matrix[i][j]+matrix[i-k][j-k]-matrix[i-k][j]-matrix[i][j-k];
            maxSum = max(maxSum, sum[i][j]);
        }
    }
    
    for (int i = k-1; i<height; i++) {
        for (int j = k-1; j<width; j++) {
            printf("%d ",sum[i][j]);
        }
        printf("\n");
    }
    
    return maxSum;
}

//363. 接雨水
//这个是标准解，两指针左右逼近。这题很好的说明了：只有最优解才有意义。最优解包含对问题的优化思路，代表着更深层的理解。
int trapRainWater(vector<int>& height) {
    if (height.size() < 2) {
        return 0;
    }
    
    int result = 0;
    int level = 0;
    bool leftSmaller = false;
    int i = 0, j = (int)height.size()-1;
    if (height[i] < height[j]) {
        leftSmaller = true;
        level = height[i];
    }else{
        leftSmaller = false;
        level = height[j];
    }
    
    while (i<j-1) {
        if (leftSmaller) {
            i++;
            result += max(0, level-height[i]);
        }else{
            j--;
            result += max(0, level-height[j]);
        }
        
        if (height[i] < height[j]) {
            level = max(height[i], level);
            leftSmaller = true;
        }else{
            level = max(height[j], level);
            leftSmaller = false;
        }
        
        //        printf("[%d, %d](%d,%d) result:%d, level:%d \n",i,j,height[i],height[j],result,level);
    }
    
    return result;
}

struct RainCell{
    int x;
    int y;
    int height;
    
    static inline int RainCellCompareFunc(RainCell &cell1, RainCell &cell2){
        if (cell1.height < cell2.height){
            return -1;
        }else if (cell1.height > cell2.height){
            return 1;
        }else{
            return 0;
        }
    }
    
    friend ostream& operator<<(ostream& os, RainCell &cell){
        os<<"("<<cell.x<<","<<cell.y<<","<<cell.height<<") ";
        return os;
    }
};

inline int handleRainCell(int x, int y, int level, TFDataStruct::heap<RainCell> &walls, vector<vector<int>> &heights, int &newCount){
    if (x<0 || y < 0 || x >= heights.size() || y >= heights.front().size()){
        return 0;
    }
    int height = heights[x][y];
    if (height < 0) {
        return 0;
    }
    heights[x][y] = -1;
    newCount++;
    
    walls.append({x, y, height});
    return max(level-height, 0);
}

//364. 接雨水II
//从第一版得到灵感，维持一个围墙是已经计算过雨水高度的，最矮位置相邻的雨水就可以立马计算出来，就是当前的雨水高度。然后把这个新的cell加入，形成一个新的围墙，而且这个围墙是缩小的。最后可以把所有cell算完。

int trapRainWater(vector<vector<int>> &heights) {
    int m = (int)heights.size(), n = (int)heights.front().size();
    if (m < 2 || n < 2){
        return 0;
    }
    TFDataStruct::heap<RainCell> walls(RainCell::RainCellCompareFunc, m*n);
    
    for (int i = 0; i<m; i++) {
        walls.append({i, 0, heights[i][0]});
        heights[i][0] = -1; //标记为负数，表示已经处理过了
        
        walls.append({i, n-1, heights[i][n-1]});
        heights[i][n-1] = -1;
    }
    for (int i = 1; i<n-1; i++) {
        walls.append({0, i, heights[0][i]});
        heights[0][i] = -1; //标记为负数，表示已经处理过了
        
        walls.append({m-1, i, heights[m-1][i]});
        heights[m-1][i] = -1;
    }
    
    auto minCell = walls.popTop();
    int level = minCell.height;
    int result = 0;
    int visitedCount = 2*(m+n-2);
    
    //1. 如果有一个cell并不是围墙的边缘，即它的四周全是访问过的cell，并不影响计算。
    //2. 如果围墙分化为多个区域，也不影响计算，因为整体的最矮cell还是它那个区域里的最小cell,而它相邻的cell也会是这个区域里的，那么计算这些相邻的cell就不会有错
    while (1) {
        
        result += handleRainCell(minCell.x-1, minCell.y, level, walls, heights, visitedCount);
        result += handleRainCell(minCell.x+1, minCell.y, level, walls, heights, visitedCount);
        result += handleRainCell(minCell.x, minCell.y-1, level, walls, heights, visitedCount);
        result += handleRainCell(minCell.x, minCell.y+1, level, walls, heights, visitedCount);
        
        if (minCell.x == 1 && minCell.y == 2) {
            cout<<walls<<endl;
        }
        printf("[%d,%d] %02d result: %d level: %d\n",minCell.x,minCell.y,minCell.height, result, level);
        
        if (visitedCount == m*n) {
            break;
        }
        minCell = walls.popTop();
        level = max(level, minCell.height);
    }
    
    return result;
}

class ExpressionTreeNode {
public:
    string symbol;
    ExpressionTreeNode *left, *right;
    ExpressionTreeNode(string symbol) {
        this->symbol = symbol;
        this->left = this->right = NULL;
    }
};

//表达式解析的级别，数值和级别高低一致
enum ExpResolveLevel{
    ExpResolveLevelAddSubtract, //加减
    ExpResolveLevelMultiDevide, //乘除
};

#define ExpressionTreeLinkRight \
if (right) {    \
opera->left = left; \
opera->right = right;   \
left = opera;   \
right = nullptr;    \
}

//TODO: 可以用非递归，即深度搜索的方式实现一遍
ExpressionTreeNode * build(vector<string> &expression, int start, int &stop, ExpressionTreeNode *startNode = nullptr) {
    
    ExpressionTreeNode *left = startNode, *right = nullptr;
    ExpressionTreeNode *opera = nullptr;
    ExpResolveLevel expLevel = ExpResolveLevelAddSubtract;
    bool inBracket = false;
    if (expression[start].front() == '(') {
        inBracket = true;
        start++;
    }
    
    while (start < expression.size()) {
        string &curStr = expression[start];
        if (curStr.front()>='0' && curStr.front()<='9') {
            if (left == nullptr) {
                left = new ExpressionTreeNode(curStr);
            }else{
                right = new ExpressionTreeNode(curStr);
            }
        }else if (curStr.front() == '+' || curStr.front() == '-'){
            if (expLevel == ExpResolveLevelMultiDevide && startNode) {
                start--;
                break;
            }
            ExpressionTreeLinkRight
            opera = new ExpressionTreeNode(curStr);
            expLevel = ExpResolveLevelAddSubtract;
            
        }else if (curStr.front() == '*' || curStr.front() == '/'){
            
            if (right) {
                //前面是加法，所以暂时没连接right,保留到现在，现在是乘法，所以这个right并不对，要继续合并后面的
                int stop = 0;
                right = build(expression, start, stop, right);
                start = stop;
            }else{
                opera = new ExpressionTreeNode(curStr);
                expLevel = ExpResolveLevelMultiDevide;
            }
            
        }else if (curStr.front() == ')'){
            ExpressionTreeLinkRight
            if (!inBracket) {
                //当前这组不是括号开始，那么当前结束点是前一个字符，这个结束括号留给上一级
                start--;
            }
            break;
        }else if (curStr.front() == '('){
            int stop = 0;
            auto subNode = build(expression, start, stop, right);
            if (!left) {
                left = subNode;
            }else{
                right = subNode;
            }
            start = stop;
        }
        
        if (expLevel == ExpResolveLevelMultiDevide) {
            ExpressionTreeLinkRight
        }
        
        start++;
    }
    if (start >= expression.size()) {
        ExpressionTreeLinkRight;
    }
    
    stop = start;
    
    return left;
}

ExpressionTreeNode * build(vector<string> &expression) {
    expression.insert(expression.begin(), "(");
    expression.push_back(")");
    int stop;
    return build(expression, 0, stop);
}

//递归 1546. 零钱问题
//非递归的数组形式，数据表太大了
int coinProblem_re(int target, vector<int>& options, int start){
    if (start == options.size()-1) {
        return target;
    }
    
    int first = 0, left = target;
    int minCount = INT_MAX;
    while (left > 0) {
        int count = first+coinProblem_re(left, options, start+1);
        minCount = min(count, minCount);
        
        left -= options[start];
        first++;
        
        if (left == 0) {
            minCount = min(count, first);
        }
    }
    
    return minCount;
}

/**
 * 1. 整体的动态规划效果都不好，不管递归还是数组存值
 * 2. 如果解里面有超过2个50，那么就可以把这2个50用一个100代替，所以50的个数最多1个，同理其他数值的最大个数也有限制，那么其他数的总和有个最大值，即50+4*20+9*10+19*5+49*2+99=512。
 3. 所以除掉512之外的部分一定是由100组成的，除掉后不足100的也要算一个100，因为这剩下的分隔其他值一定会移除。这样就可以先求出512以内的最优解，然后把目标减去100的倍数直到目标落在512以内。
 4. 最快的解法是100以内的用其他数组成，其他部分都是100组成，但是这个我无法证明。由此引发的一个问题是：什么样的值结构可以保证最大值可以“贪心”?比如值选择有：100,40,1。求总数160的分配，4个40是最快的，如果用了100，就会变成100+40+20*1,总数是22个。
 */
int coinProblem(int n, int m) {
    vector<int> options = {100,50,20,10,5,2,1};
    
    //每一个最优解都是从前面一个更小数的最有解加上一个数变过来的
    //最大数是100，所以我们只需要维持100以内的最优解就可以了
    int minZone = 512;
    int zoneResult[minZone+1];
    zoneResult[0] = 0;
    zoneResult[1] = 1;
    zoneResult[2] = 1;
    
    for (int i = 3; i<=minZone; i++) {
        int minCount = INT_MAX;
        for (int op : options) {
            if (i>=op) {
                minCount = min(minCount, zoneResult[i-op]);
            }
        }
        
        zoneResult[i] = minCount+1;
    }
    
    int target = n-m;
    if (target <= minZone) {
        return zoneResult[target];
    }else{
        int hundredCount = ceil((target-minZone)/100.0);
        return hundredCount+zoneResult[target-hundredCount*100];
    }
    
    return 0;
}

//1478. 最接近target的值
//双指针，当i-j的和小于target时，把i增加，直到和大于target(等于就直接得结果了，就不必说了)，这时上一次的和就是这个j对应的最优解
//然后把j减小一个，i也减小一个回到最优解的位置。如果i往更小的移动，那么这时的和对比上一次的最优解：i小了，j也小了，综合就小了，那么它一定比上次的最优解更远离target。所以i不能左移了，从当前开始，不断右移。这时就回到了上一步：和小于target时就i++直到和大于target.
//整个过程其实是在j固定时所对应的最优解，然后比较这些最优解得到最终解。只是每次j减小后，i不用从0开始算，而是从上一次的解的位置接着往下算。这样i和j都不会走重复的地方，所以复杂度还是O(n)
int closestTargetValue(int target, vector<int> &array) {
    sort(array.begin(), array.end());
    
    if (array[0]+array[1]>target) {
        return -1;
    }
    
    int i = 0, j = (int)array.size()-1;
    int sum;
    do {
        sum = array[i]+array[j];
        j--;
    } while (sum > target);
    
    if (sum == target) {
        return target;
    }
    
    j++;
    i++;
    
    int result = sum;
    
    while (i<j) {
        sum = array[i]+array[j];
        if (sum < target) {
            result = max(result, sum);
            i++;
        }else if (sum == target){
            return target;
        }else{
            i--;
            j--;
        }
    }
    
    return result;
}

string isTwin(string &s, string &t) {
    if (s.size() != t.size()) {
        return "No";
    }
    
    //奇数的数量统计在后216个槽位里，偶数在前面
    short size = 216;
    int count_s[size*2];
    int count_t[size*2];
    memset(count_s, 0, size*2*sizeof(int));
    memset(count_t, 0, size*2*sizeof(int));
    
    for (int i = 0; i<s.size(); i++) {
        //i&1，在奇数时为1，偶数时为0，奇数会偏移216
        count_s[s[i]+size*(i&1)]++;
    }
    for (int i = 0; i<t.size(); i++) {
        count_t[t[i]+size*(i&1)]++;
    }
    
    for (int i = 0; i<size*2; i++) {
        if (count_s[i] != count_t[i]) {
            return "No";
        }
    }
    return "Yes";
}

vector<string> convertToRPN(vector<string> &expression) {
    stack<string> op;
    vector<string> RPN;
    for (string &str : expression){
        char &fir = str.front();
        if (fir >= '0' && fir <= '9') {
            RPN.push_back(str);
        }else{
            if (fir == '(') {
                op.push(str);
            }else if (fir == ')'){
                
                while (op.top().front() != '(') {
                    RPN.push_back(op.top());
                    op.pop();
                }
                op.pop();
            }else if(fir == '+' || fir == '-'){
                while (!op.empty() && op.top().front() != '(') {
                    RPN.push_back(op.top());
                    op.pop();
                }
                op.push(str);
            }else{
                while (!op.empty() && (op.top().front() == '*' || op.top().front() == '/')) {
                    RPN.push_back(op.top());
                    op.pop();
                }
                op.push(str);
            }
        }
    }
    
    while (!op.empty()) {
        RPN.push_back(op.top());
        op.pop();
    }
    
    return RPN;
}

void convertToRPN(vector<string> &expression, vector<string> &RPN) {
    stack<string> op;
    for (string &str : expression){
        char &fir = str.front();
        if (fir >= '0' && fir <= '9') {
            RPN.push_back(str);
        }else{
            if (fir == '(') {
                op.push(str);
            }else if (fir == ')'){
                
                while (op.top().front() != '(') {
                    RPN.push_back(op.top());
                    op.pop();
                }
                op.pop();
            }else if(fir == '+' || fir == '-'){
                while (!op.empty() && op.top().front() != '(') {
                    RPN.push_back(op.top());
                    op.pop();
                }
                op.push(str);
            }else{
                while (!op.empty() && (op.top().front() == '*' || op.top().front() == '/')) {
                    RPN.push_back(op.top());
                    op.pop();
                }
                op.push(str);
            }
        }
    }
    
    while (!op.empty()) {
        RPN.push_back(op.top());
        op.pop();
    }
}

int evalRPN(vector<string> &tokens) {
    stack<int> nums;
    
    for (int i = 0; i<tokens.size(); i++) {
        char c = tokens[i].back();
        if (c >= '0' && c <= '9') {
            nums.push(stoi(tokens[i]));
        }else{
            int num1 = nums.top();
            nums.pop();
            int num2 = nums.top();
            nums.pop();
            
            int result = 0;
            if (c == '+') {
                result = num1+num2;
            }else if (c == '-'){
                result = num2-num1;
            }else if (c == '*'){
                result = num1*num2;
            }else{
                result = num2/num1;
            }
            
            nums.push(result);
        }
    }
    
    if (nums.empty()) {
        return 0;
    }
    return nums.top();
}

int evaluateExpression(vector<string> &expression) {
    vector<string> RPN;
    convertToRPN(expression, RPN);
    return evalRPN(RPN);
}

//79. 最长公共子串  动态规划
struct MaxLenNode{
    int maxLen;
    int involveHeadLen;
};

int longestCommonSubstring(string &A, string &B) {
    if (A.empty() || B.empty()) {
        return 0;
    }
    
    int w = (int)A.size(), h = (int)B.size();
    MaxLenNode maxLenMap[w][h];
    
    for (int i = 0; i<w; i++) {
        int count = (A[i]==B.back()?1:0);
        maxLenMap[i][h-1] = {count, count};
    }
    
    for (int j = 0; j<h-1; j++) {
        int count = (B[j]==A.back()?1:0);
        maxLenMap[w-1][j] = {count, count};
    }
    
    for (int i = w-2; i>=0; i--) {
        for (int j = h-2; j>=0; j--) {
            int maxLen = 0, maxInvolve = 0;
            if (A[i] == B[j]) {
                maxInvolve = maxLen = 1+maxLenMap[i+1][j+1].involveHeadLen;
            }
            
            maxLen = max(maxLen, maxLenMap[i+1][j+1].maxLen);
            maxLen = max(maxLen, maxLenMap[i+1][j].maxLen);
            maxLen = max(maxLen, maxLenMap[i][j+1].maxLen);
            
            maxLenMap[i][j] = {maxLen, maxInvolve};
        }
    }
    
    return maxLenMap[0][0].maxLen;
}

int singleNumber(vector<int> &A) {
    int num = 0;
    for (auto &n : A){
        num ^= n;
    }
    
    return num;
}

vector<int> getNarcissisticNumbers(int n) {
    if (n == 1) {
        return {1,2,3,4,5,6,7,8,9};
    }
    
    vector<int> result;
    int pows[10];
    for (int i = 0; i<10; i++) {
        pows[i] = pow(i, n);
    }
    
    int min = pow(10, n-1);
    int max = pow(10, n);
    int digits[n];
    memset(digits, 0, sizeof(digits));
    digits[0] = 1;
    
    //除法特别慢，计算机没法优化，用加法
    for (int k = min+1; k<max; k++) {
        bool carry = 1;
        for (int i = n-1; i>=0 && carry; i--) {
            if (digits[i] == 9) {
                digits[i] = 0;
            }else{
                digits[i]++;
                carry = 0;
            }
        }
        
        int powSum = 0;
        for (int i = 0; i<n; i++) {
            powSum += pows[digits[i]];
        }
        
        if (powSum == k) {
            result.push_back(powSum);
        }
    }
    
    return result;
}

///1与，对方；1或，都是1；0与，都是0；0或，对方
int aplusbx(int a, int b) {
    int result = 0;
    bool carry = 0;
    for (int i = 0; i<sizeof(int)*8; i++) {
        int digitA = a &1;
        int digitB = b &1;
        
        bool one = false;
        if (digitA ^ digitB) {  //和为1
            if (!carry) {
                one = true;
            }
        }else{  //和为0
            if (digitA) {
                if (carry) {
                    one = true;
                }
                carry = true;
            }else{
                if (carry) {
                    one = true;
                    carry = false;
                }
            }
        }
        
        if (one) {
            result |= (1<<i);
        }
        
        a >>= 1;
        b >>= 1;
    }
    
    return result;
}

void rotateString(string &str, int start, int end){
    int i = start, j = end;
    while (i<j) {
        char temp = str[i];
        str[i] = str[j];
        str[j] = temp;
        
        i++;
        j--;
    }
}

void rotateString(string &str, int offset) {
    int len = (int)str.length();
    if (len == 0) {
        return;
    }
    offset %= len;
    if (offset == 0) {
        return;
    }
    
    rotateString(str, 0, len-offset-1);
    rotateString(str, len-offset, len-1);
    rotateString(str, 0, len-1);
}


vector<int> twoSum(vector<int> &numbers, int target) {
    
    vector<int> result;
    unordered_map<int, int> hash;
    for (int i = 0; i < numbers.size(); ++i) {
        if (hash.find(target - numbers[i]) != hash.end()) {
            result.push_back(hash[target - numbers[i]]);
            result.push_back(i);
            break;
        }
        hash[numbers[i]] = i;
    }
    
    return result;
}

static int XORMerge(int &a, int &b){
    return a^b;
}

vector<int> intervalXOR(vector<int> &A, vector<Interval> &query) {
    typedef TFDataStruct::SegmentTree<int, XORMerge> SegmentTree;
    auto root = SegmentTree::build(A);
    vector<int> result;
    for (auto &qu : query){
        result.push_back(SegmentTree::query(root, qu.start, qu.end+qu.start-1));
    }
    
    return result;
}

//q(n*a1-n+n*(n-1)/2)
int getSum(int A, int B) {
    int start = (A+2)/3;
    int end = B/3;
    
    int count = end-start+1;
    return 3*(count*start+count*(count-1)/2);
}

char firstUniqChar(string &str) {
    unordered_map<char, int> mark;
    for (auto &c : str){
        mark[c]++;
    }
    
    for (auto &c : str){
        if (mark[c] == 1) {
            return c;
        }
    }
    
    return '\0';
}

bool isUnique(string &str) {
    bool mark[256];
    memset(mark, 0, sizeof(mark));
    
    for (auto &c : str){
        if (mark[c]) {
            return false;
        }else{
            mark[c] = true;
        }
    }
    
    return true;
}

int strStr(string &source, string &target) {
    int len1 = (int)source.length(), len2 = (int)target.length();
    for (int x = 0; x<=len1-len2; x++) {
        int i = x, j = 0;
        while (i< len1 && j<len2 && source[i] == target[j]) {
            i++;
            j++;
        }
        if (j == len2) {
            return x;
        }
    }
    return -1;
}

bool isLegalIdentifier(string &str) {
    if (str.length() == 0) {
        return true;
    }
    if (str.front()>='0' && str.front() <= '9') {
        return false;
    }
    
    for (auto &c : str){
        if (c =='_' ||
            (c<='9' && c >= '0') ||
            (c<='z' && c >= 'a') ||
            (c<='Z' && c >= 'A')) {
            continue;
        }else{
            return false;
        }
    }
    
    return true;
}

void sortIntegers(vector<int> &A) {
    for (int i = 1; i<A.size(); i++) {
        int cur = A[i];
        int j = i-1;
        while (j>=0 && A[j]>cur) {
            A[j+1] = A[j];
            j--;
        }
        A[j+1] = cur;
    }
}

static bool intervalComp(Interval &a, Interval &b){
    return a.start < b.start;
}
bool canAttendMeetings(vector<Interval> &intervals) {
    sort(intervals.begin(), intervals.end(), intervalComp);
    
    auto &lastInterval = intervals.front();
    for (int i = 1; i< intervals.size(); i++){
        if (intervals[i].start<=lastInterval.end) {
            return false;
        }
        lastInterval = intervals[i];
    }
    
    return true;
}

int deduplication(vector<int> &nums) {
    if (nums.size() < 2) {
        return (int)nums.size();
    }
    
    unordered_map<int, bool> exist;
    exist[nums.front()] = true;
    
    int i = 1, j = 1;
    while (j < nums.size()) {
        if (exist.find(nums[j]) == exist.end()) {
            nums[i]=nums[j];
            exist[nums[j]] = true;
            i++;
        }
        j++;
    }
    
    return i;
}

///最小堆，然后求和
string depress(int m, int k, vector<int> &arr) {
    //维持k个最小值s，使用最大堆
    TFDataStruct::heap<int> minHeap(false, k);
    
    for (auto &num : arr){
        if (!minHeap.isFull()) {
            minHeap.append(num);
        }else{
            cout<<minHeap<<endl;
            if (num < minHeap.getTop()) {
                minHeap.replaceTop(num); //跟新顶部
            }
        }
    }
    
    cout<<minHeap<<endl;
    
    int sum = 0;
    while (!minHeap.isEmpty()) {
        sum += minHeap.popTop();
    }
    
    return sum<m?"yes":"no";
}



//思路归纳： 一个查找区间，目标确定在这个区间内，将区间划分成两部分，判定区间在左边还是在右边，然后就可以缩小查找区间。循环不变体是：目标在查找区间内；变化过程是：区间的二分切割；退出条件时：区间缩小到长度为1.
//再高一层的抽象为：大问题化小，小到一定程度，问题便非常简单可解了。
int getAns(vector<int> &a) {
    if (a.empty()) {
        return -1;
    }
    int mid = (int)(a.size()-1)/2;
    vector<int> copy = a;
    int left = 0, right = (int)a.size()-1;
    while (left < right) {
        int k = partion(copy, left, right);
        printVectorOneLine(copy);
        printf("k=%d\n",k);
        if (k==mid) {
            left = right = k;
            break;
        }else if (k < mid) {
            left = k+1;
        }else{
            right = k-1;
        }
    }
    
    int find = copy[left];
    for (int i = 0; i<a.size(); i++) {
        if (a[i] == find) {
            return i;
        }
    }
    
    return -1;
}

int maximumSwap(int num) {
    if (num < 10) {
        return num;
    }
    vector<int> digits;
    int num2 = num;
    while (num2 > 0) {
        digits.push_back(num2%10);
        num2 /= 10;
    }
    
    int maxIdx = 0, maxNum = digits.front();
    int change1 = -1, change2 = -1;
    for (int i = 1; i<digits.size(); i++) {
        if (digits[i]>maxNum) {
            maxNum = digits[i];
            maxIdx = i;
        }else if (digits[i] < maxNum){
            change1 = maxIdx;
            change2 = i;
        }
    }
    
    if (change1 >= 0) {
        int temp = digits[change1];
        digits[change1] = digits[change2];
        digits[change2] = temp;
    }
    
    int result = 0, weight = 1;
    for (auto &d : digits){
        result += d*weight;
        weight *= 10;
    }
    
    return result;
}

int countNodes(ListNode * head) {
    int count = 0;
    ListNode *cur = head;
    while (cur) {
        count++;
        cur = cur->next;
    }
    
    return count;
}

ListNode * middleNode(ListNode * head) {
    if (head == nullptr) {
        return nullptr;
    }
    
    ListNode *slow = head, *fast = head;
    bool go = true;
    while (1) {
        fast = fast->next;
        if (!fast) {
            break;
        }
        go = !go;
        if (go) {
            slow = slow->next;
        }
    }
    
    return slow;
}

ListNode * insertNode(ListNode * head, int val) {
    
    ListNode *newNode = new ListNode(val);
    if (head == nullptr) {
        return newNode;
    }
    
    ListNode *cur = head, *last = nullptr;
    while (cur && cur->val < val) {
        last = cur;
        cur = cur->next;
    }
    
    if (last) {
        last->next = newNode;
    }else{
        head = newNode;
    }
    newNode->next = cur;
    
    return head;
}

vector<int> reverseStore(ListNode * head) {
    vector<int> result;
    ListNode *cur = head;
    while (cur) {
        result.insert(result.begin(), cur->val);
        cur = cur->next;
    }
    
    return result;
}

ListNode * deleteNode(ListNode * head, int n, int m) {
    ListNode *pre = nullptr, *after = nullptr, *cur = head;
    int index = 0;
    while (cur) {
        
        if (index == n-1) {
            pre = cur;
        }else if (index == m){
            after = cur->next;
            break;
        }
        
        index++;
        cur = cur->next;
    }
    
    if (pre == nullptr) {
        return after;
    }
    pre->next = after;
    return pre;
}

bool findHer(vector<string> &maze, Point start, Point end){
    if (maze[start.x][start.y] == '*' ||
        start.x<0||
        start.x >= maze.size() ||
        start.y <0||
        start.y >= maze.front().size()) {
        return false;
    }
    
    if (start.x == end.x && start.y == end.y) {
        return true;
    }
    
    maze[start.x][start.y] = '*';
    if (findHer(maze, {start.x+1, start.y}, end)) {
        return true;
    }
    if (findHer(maze, {start.x-1, start.y}, end)) {
        return true;
    }
    if (findHer(maze, {start.x, start.y+1}, end)) {
        return true;
    }
    if (findHer(maze, {start.x, start.y-1}, end)) {
        return true;
    }
    //    maze[start.x][start.y] = '.';
    
    return false;
}

class MyStringHashMap{
    struct ListNode {
    public:
        int val = 0;
        string key;
        ListNode *next;
        ListNode(string &key) {
            this->key = key;
            this->next = NULL;
        }
    };
    
    int size;
    ListNode *data;
    hash<string> hashCreator;
    
    int visitCount = 0;
    long long visitStep = 0;
    
    ListNode *findNodeWithKey(string &key){
        visitCount++;
        auto pos = hashCreator(key)%size;
        ListNode &head = data[pos];
        
        ListNode *cur = head.next, *last = &head;
        visitStep++;
        while (cur && cur->key.compare(key) != 0) {
            last = cur;
            cur = cur->next;
            visitStep++;
        }
        
        //        if (cur == nullptr) {
        //            cur = new ListNode(key);
        //        }else{
        //            last->next = cur->next;
        //        }
        //        //被访问的都插到开头，采用类似LRU的策略
        //        cur->next = head.next;
        //        head.next = cur;
        
        if (cur == nullptr) {
            cur = new ListNode(key);
            last->next = cur;
        }
        
        return cur;
    }
    
public:
    MyStringHashMap(int size):size(size){
        data = (ListNode*)malloc(size*sizeof(ListNode));
        memset(data, 0, sizeof(ListNode)*size);
    };
    
    int& operator[](string &key){
        ListNode *find = findNodeWithKey(key);
        return find->val;
    }
};

vector<int> getAns(vector<int> &op, vector<string> &name, vector<int> &w) {
    //    unordered_map<string, int> accounts;
    int size = (int)op.size();
    MyStringHashMap accounts(size/2);
    
    vector<int> result(op.size(), 0);
    for (int i = 0; i<size; i++) {
        string &key = name[i];
        int m = accounts[key];
        if (!op[i]) {
            result[i] = accounts[key] = m+w[i];
        }else{
            if (m < w[i]) {
                result[i] = -1;
            }else{
                result[i] = accounts[key] = m-w[i];
            }
        }
    }
    
    return result;
}

bool findHer(vector<string> &maze) {
    
    Point start, end;
    
    int x = 0, y = 0;
    for (auto & str : maze){
        y = 0;
        for (auto &c : str){
            if (c == 'S') {
                start = {x, y};
            }else if (c == 'T'){
                end = {x,y};
            }
            y++;
        }
        x++;
    }
    
    return findHer(maze, start, end);
}


///1. 通过范围来避免重复  2. 检查每个参数的定义和它们的传值是否正确 3. 审题，输出结果是否是题目需求
void getWays(vector<int> &a, int start, int k, int sum, int &count) {
    
    if (k == 0) {
        if (isPrimeNum(sum)) {
            count++;
        }
        return;
    }
    
    for (int i = start; i<a.size(); i++) {
        getWays(a, i+1, k-1, sum+a[i], count);
    }
}

int getWays(vector<int> &a, int k) {
    int count = 0;
    getWays(a, 0, k, 0, count);
    
    return count;
}

long long getAnsxx(vector<int> &atk) {
    sort(atk.begin(), atk.end());
    
    int sum = 0;
    int idx = 0;
    for (int k = atk.size()-1; k>0; k--) {
        sum += k*atk[idx];
    }
    
    return sum;
}

//使用贪心算法的问题在于：类似 9 7 8 9，如果给两次机会，那么贪心(9+(7+8)) 9结果24、9
//而实际更好的解答是(9+7)(8+9)，结果16 17
//以最小值为核心的贪心策略是不成立的，因为需要考虑到以后的变化，这个就和贪心只考虑局部的主旨冲突了
int getDistance(int n, int m, int target, vector<int> &d) {
    
    vector<pair<int, int>> gaps;
    vector<int> gaps2;
    int pre = 0;
    for (int i = 0; i<d.size(); i++) {
        gaps.push_back({d[i]-pre, i});
        gaps2.push_back(d[i]-pre);
        pre = d[i];
    }
    gaps.push_back({target-pre, n});
    
    
    
    for (int i = 0; i<m; i++) {
        int minIdx = -1, minVal = INT_MAX;
        for (int j = 0; j<gaps.size(); j++) {
            if (gaps[j].first < minVal) {
                minVal = gaps[j].first;
                minIdx = j;
            }
        }
        
        int left = INT_MAX,right = INT_MAX;
        if (minIdx>0) {
            left = gaps[minIdx-1].first;
        }
        if (minIdx<gaps.size()-1) {
            right = gaps[minIdx+1].first;
        }
        
        printf("移除 %d\n",gaps[minIdx].second);
        
        if (left < right) {
            gaps[minIdx].first += gaps[minIdx-1].first;
            gaps.erase(gaps.begin()+minIdx-1);
            //            printf("融合%d, %d + %d= %d[%d]\n",minIdx-1,left,minVal,left+minVal,minVal);
        }else{
            gaps[minIdx+1].first += gaps[minIdx].first;
            gaps.erase(gaps.begin()+minIdx);
            //            printf("融合%d, %d + %d= %d[%d]\n",minIdx,minVal,right,right+minVal,minVal);
        }
        
        //        printVectorOneLine(gaps);
    }
    
    int minVal = INT_MAX;
    for (int j = 0; j<gaps.size(); j++) {
        if (gaps[j].first < minVal) {
            minVal = gaps[j].first;
        }
    }
    
    bool save[d.size()];
    memset(save, 0, sizeof(save));
    for (auto &p : gaps){
        printf("保留 %d\n",p.second);
        save[p.second] = true;
    }
    
    for (int i = 0; i<gaps2.size(); i++) {
        
        if (save[i]) {
            printf(" %d ",gaps2[i]);
        }else{
            printf(" [%d] ",gaps2[i]);
        }
    }
    
    return minVal;
}

int moveCount(vector<int> &d, int minGap){
    int count = 0;
    int pre = 0;
    for (int i = 0; i<d.size(); i++) {
        if (d[i]-pre<minGap) {
            count++;
        }else{
            pre = d[i];
        }
    }
    
    return count;
}

void showGaps(vector<int> &d, int minGap){
    int pre = 0;
    for (int i = 0; i<d.size(); i++) {
        if (d[i]-pre<minGap) {
            printf(" [%d] ",d[i]-(i==0?pre:d[i-1]));
        }else{
            printf(" %d ",d[i]-(i==0?pre:d[i-1]));
            pre = d[i];
        }
    }
    
    printf("\n");
    pre = 0;
    for (int i = 0; i<d.size(); i++) {
        if (d[i]-pre>minGap) {
            printf("%d ",d[i]-pre);
            pre = d[i];
        }
        
    }
}

int getDistance2(int n, int m, int target, vector<int> &d){
    
    
    //从左到右，距离递增，次数也递增；左侧count小，则条件左侧为count<=m,右侧为count>m
    //[i,j]都是考察范围，范围以外是左右侧
    int i = 0, j = target;
    while (i<=j) {
        int mid = i+(j-i)/2;
        int count = moveCount(d, mid);
        if (count <= m) {
            i = mid+1;
        }else{
            j = mid-1;
        }
    }
    
    showGaps(d, j);
    
    return j;
}

/*
 如果没局大家都是玩家，那么结果是固定的，现在每局有一个人是裁判，那么问题就转为对裁判的分配问题。
 现在采取的策略是谁的玩家次数有多余，就给谁分配裁判。如果裁判数量多余，那么肯定是失败的。因为多余的裁判分配到某个人身上，它的玩家次数就不满足自己的期望了，就失败。
 问题是裁判可以全部分配完，这样的分配就一定具有可行性吗？临界情况就是每个人当玩家的次数刚好是自己的期望。
 然而，这个游戏过程没有任何的阻碍，分配好裁判后，就一定可以执行到结果，根本不存在不可行的问题。
 */

/*
 1. 当游戏数固定的时候，裁判的分配才会影响最后的结果，分配完，执行的顺序不影响结果。
 2. 纠结执行顺序是因为我们希望更快的消灭"当玩家"的期望，这是一种线性的思维，导致我们会去在意当前每一步的操作。这里总结就是“贪心结果是一样的”
 3. 二分法把次数变成了定量，而满足"当玩家"的期望的程度变成了变量，而题目本身是期望是定量（也就是得到满足），而数量是变量。二分法直接把思维的方向给调换了。
 4. 之所以可以用二分法做这样的调换，在于这两个变量之间的单调性，即次数越多，满足程度就越大，这样两者之间是一种确定的关系，从而才可以做推倒。从这一点出发，倘若两者之间是一个函数y=f(x)的复杂关系，如果可以加入新的变量/定义，形成行的关系x1,y1,是的它们h之间是单调关系，那么也是可行的。
 */
bool canGameOver(vector<int> &A, long long gameCount){
    long long freeCount = 0;
    for (auto &c : A){
        freeCount += gameCount-c;
    }
    
    return freeCount>=gameCount;
}

/*
 用了二分法的思路，被二分的考察数据是玩游戏的次数。二分的核心问题之一就是如何快速判定中位数时的解的情况，从而拆分考察范围。
 这里就是：执行了k次游戏时，是否可以满足所有人当玩家的期望
 */
long long playGames(vector<int> &A) {
    int maxVal = -1;
    for (int i = 0; i<A.size(); i++) {
        if (A[i]>maxVal) {
            maxVal = A[i];
        }
    }
    
    long long i = maxVal, j = maxVal*2;
    //[i,j)
    while (i<j) {
        long long mid = i+(j-i)/2;
        if (canGameOver(A, mid)) {
            j = mid;
        }else{
            i = mid+1;
        }
    }
    
    return j;
}

//bool roundGame(vector<int> &A){
//    int minVal = INT_MAX, minIdx = -1;
//    for (int i = 0; i<A.size(); i++) {
//        if (A[i]<minVal) {
//            minVal = A[i];
//            minIdx = i;
//        }
//        A[i]--;
//    }
//
//    A[minIdx]++;
//
//
//    for (auto &c : A){
//        if (c > 0) {
//            return false;
//        }
//    }
//
//    return true;
//}

long long playGames1(vector<int> &A) {
    int count = 0;
    while (1) {
        sort(A.begin(), A.end());
        if (A.front() <= 0) {
            return count + A.back();
        }
        int count1 = A[1]-A[0]+1;
        for (int i = 1; i<A.size(); i++) {
            A[i] -= count1;
        }
        count += count1;
    }
    
    return count;
}

long long playGames2(vector<int> &A) {
    
    if (A.size() == 2) {
        return A[0]+A[1];
    }
    
    int maxVal = -1;
    for (int i = 0; i<A.size(); i++) {
        if (A[i]>maxVal) {
            maxVal = A[i];
        }
    }
    
    return maxVal;
}

template<class T>
int binaryFindLower(vector<T> &nums, T target){
    int i = -1, j = (int)nums.size();  //左 <, 右 >,边界不包含
    while (i<j-1) {
        int mid = i+(j-i)/2;
        if (nums[mid] == target) {
            return mid;
        }else if (nums[mid] < target){
            i = mid;
        }else{
            j = mid;
        }
    }
    return i;
}

long long doingHomework(vector<int> &cost, vector<int> &val) {
    vector<long long> calculateCost;
    int pre = 0;
    for (auto &c : cost){
        pre += c;
        calculateCost.push_back(pre);
    }
    
    printVectorOneLine(calculateCost);
    
    long long sum = 0;
    for (int v : val){
        int idx = binaryFindLower(calculateCost, (long long)v);
        if (idx >= 0) {
            //            printf("%d --> %lld \n",v,calculateCost[idx]);
            sum += calculateCost[idx];
        }
    }
    
    return sum;
}

/*
 思考了之后，发现跟“玩游戏”那一题竟然同一个模型。
 都是俄罗斯方块消除的模型：那一题消除的是每个人玩游戏的期望，这题消除的是任务。
 如果都看成是任务：那么那题是每一轮有一个任务不做，其他的都做，总数时人数；这题是选出n个任务做,n是相同任务之间的空隙+1，就是周期的长度。区别只是这一点，目标都是球把所有任务做完的最短次数。
 但是这一题似乎无法用二分法，很难快速的从次数直接得到是否成功，这个是拆分区间的关键。
 */
int leastInterval(string &tasks, int n) {
    vector<int> heights(26,0);
    
    for (auto &c : tasks){
        heights[c-'A']++;
    }
    
    n += 1;
    int totalCount = 0;
    
    if (n < 26) {
        do {
            sort(heights.begin(), heights.end());
            int first = heights[26-n];
            if (first <= 0) {
                totalCount += heights.back();
                break;
            }
            totalCount += first;
            
            for (int i = 26-n; i<heights.size(); i++) {
                heights[i] -= first;
            }
            
        } while (1);
    }else{
        sort(heights.begin(), heights.end());
        totalCount += heights.back();
    }
    
    int lastRowCount = 0, maxVal = heights.back();
    for (int i = 25; i>=0; i--) {
        if (heights[i] == maxVal) {
            lastRowCount++;
        }else{
            break;
        }
    }
    
    totalCount *= n;
    if (lastRowCount<n) {
        totalCount -= (n-lastRowCount);
    }
    
    return totalCount;
}

int stackSorting(stack<int> &stk) {
    stack<int> stk2;
    
    while (!stk.empty()) {
        int top = stk.top();
        stk.pop();
        
        int backCount = 0;
        while (!stk2.empty() && stk2.top()>top) {
            stk.push(stk2.top());
            stk2.pop();
            backCount++;
        }
        
        stk2.push(top);
        for (int i = 0; i<backCount; i++) {
            stk2.push(stk.top());
            stk.pop();
        }
    }
    
    stk = stk2;
    
    return 0;
}

class MyQueue {
    stack<int> stack1;
    stack<int> stack2;
    
    void move(){
        while (!stack1.empty()) {
            stack2.push(stack1.top());
            stack1.pop();
        }
    }
    
public:
    MyQueue() {
        
    }
    
    void push(int element) {
        stack1.push(element);
    }
    
    int pop() {
        if (stack2.empty()) {
            move();
        }
        int val = stack2.top();
        stack2.pop();
        return val;
    }
    
    int top() {
        if (stack2.empty()) {
            move();
        }
        return stack2.top();
    }
};

/** 验证出栈序列的合法性，如果合法，则返回栈内数量最多是的个数，否则返回0 */
//我使用的判断逻辑是：元素pop(i)之后所有小于它的元素都是单调递减的。这时的pop指的是入栈序列构成的出栈序列，而不是值本身。
int isLegalSeq(vector<int> &popIndexes){
    
    int count = (int)popIndexes.size();
    bool checked[count];
    memset(checked, 0, sizeof(checked));
    
    int maxCount = 0;
    for (int i = 0; i<count; i++) {
        if (checked[i]) {
            continue;
        }
        
        int lessCount = 1;
        int ref = popIndexes[i], last = ref;
        for (int j = i+1; j<count; j++) {
            int cur = popIndexes[j];
            if (cur < ref) {
                if (cur>last) {
                    return -1;
                }else{
                    last = cur;
                    checked[j] = true;
                    lessCount++;
                }
            }
        }
        
        if (lessCount > maxCount) {
            maxCount = lessCount;
        }
    }
    
    return maxCount;
}

bool isLegalSeq(vector<int> &push, vector<int> &pop) {
    //    unordered_map<int, vector<int>> pushIndexes;
    //    for(int i = 0; i<push.size(); i++){
    //        if (pushIndexes.find(push[i]) == pushIndexes.end()) {
    //            pushIndexes[push[i]] = {i};
    //        }else{
    //            pushIndexes[push[i]].push_back(i);
    //        }
    //    }
    
    //不考虑重复问题
    unordered_map<int, int> pushIndexes;
    for(int i = 0; i<push.size(); i++){
        pushIndexes[push[i]] = i;
    }
    
    int count = (int)pop.size();
    vector<int> popIndexes(count, 0);
    for (int i = 0; i<count; i++) {
        popIndexes[i] = pushIndexes[pop[i]];
    }
    
    return isLegalSeq(popIndexes)>=0;
}

int trainCompartmentProblem(vector<int> &arr) {
    int maxCount = isLegalSeq(arr);
    if (maxCount>0) {
        maxCount -= 1;
    }
    return maxCount;
}

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
    
    typedef TFDataStruct::MyDirectedGraph<int> GraphType;
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

//先cost，再city1，再city2
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
    return result<0;
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

//629. 最小生成树
//1. 边的选择要排序， 内部用索引排序，所以外部先按照city的排序规则把city排序
vector<Connection> lowestCost_mat(vector<Connection>& connections) {
    typedef TFDataStruct::DirectedGraph<string> GraphType;
    GraphType graph;
    
    //使用map来做去重，同时也为了后面让city和索引建立映射
    map<string, int> cityIdx;
    for (auto &con : connections){
        if (cityIdx.find(con.city1) == cityIdx.end()) {
            cityIdx[con.city1] = 1;
        }
        if (cityIdx.find(con.city2) == cityIdx.end()) {
            cityIdx[con.city2] = 1;
        }
    }
    
    //使用邻接矩阵保存
    graph.initMatrix((int)cityIdx.size());
    
    int index = 0;
    for (auto &pair : cityIdx) {
        graph.allNodes.push_back(GraphType::NodeType(pair.first));
        pair.second = index;
        index++;
    }
    
    for (auto &con : connections){
        int edgeIdx1 = cityIdx[con.city1];
        int edgeIdx2 = cityIdx[con.city2];
        
        graph.matrix[edgeIdx1][edgeIdx2] = min(con.cost, graph.matrix[edgeIdx1][edgeIdx2]);
    }
    
    auto edges = graph.lowestTree_kruskal();
    vector<Connection> result;
    for (auto &edge : edges) {
        
        string &val1 = graph.allNodes[edge.first].val;
        string &val2 = graph.allNodes[edge.second].val;
        result.push_back({val1, val2, edge.cost});
    }
    
    sort(result.begin(), result.end(), connectComp);
    
    return result;
}

bool textComp(string &str1, string &str2){
    int result = str1.compare(str2);
    if (result<=0) {
        return true;
    }else{
        return false;
    }
}

bool numsComp(int &n1, int &n2){
    return n1<=n2;
}

static bool pointComp(Point &p1, Point &p2){
    return p1.x<p2.x;
}

vector<int> getPoints(vector<Point> &points) {
    unordered_map<int, int> idxMap;
    for (int i = 0; i<points.size(); i++) {
        idxMap[points[i].x] = i;
    }
    
    sort(points.begin(), points.end(), pointComp);
    
    int first = -1, second = -1;
    float maxSlope = 0;
    for (int i = 0; i<points.size()-1; i++) {
        float slope = (points[i+1].y-points[i].y)/(points[i+1].x-points[i].x);
        if (slope>maxSlope) {
            first = idxMap[points[i].x];
            second = idxMap[points[i+1].x];
            if (first>second) {
                int temp = first;
                first = second;
                second = temp;
            }
            maxSlope = slope;
        }else if (slope == maxSlope){
            int first2 = idxMap[points[i].x];
            int second2 = idxMap[points[i+1].x];
            if (first2>second2) {
                int temp = first2;
                first2 = second2;
                second2 = temp;
            }
            if (first2<first || (first == first2 && second2<second)) {
                first=first2;
                second = second2;
            }
        }
    }
    
    return {first,second};
}

vector<int> subarraySumClosest(vector<int> &nums) {
    vector<pair<int, int>> preSums;
    int sum = 0;
    for (int i = 0; i<nums.size(); i++) {
        sum += nums[i];
        preSums.push_back({sum,i});
    }
    
    sort(preSums.begin(), preSums.end(), PairSort<int, int>::pairFirstComp);
    
    int minDis = INT_MAX, first=-1, second=0;
    for (int i = 1; i<preSums.size(); i++) {
        int dis = preSums[i].first-preSums[i-1].first;
        if (dis<minDis) {
            minDis = dis;
            first = preSums[i-1].second;
            second = preSums[i].second;
            
            FirstSmall(first, second)
        }else if (dis == minDis){
            int first2 = preSums[i-1].second;
            int second2 = preSums[i].second;
            
            FirstSmall(first2, second2)
            if (first2<first || (first == first2 && second2<second)) {
                first = first2;
                second = second2;
            }
        }
    }
    
    return {first+1,second};
}

class Comparator {
public:
    int cmp(string a, string b){
        transform(a.begin(), a.end(), a.begin(), ::toupper);
        transform(b.begin(), b.end(), b.begin(), ::toupper);
        return a.compare(b);
    }
};

int NutsAndBoltsPartion(vector<string> &vals, string &stan, Comparator &compare, int start, int end, bool stanFirst){
    int i = start, j = end;
    bool left = true;
    int equalIdx = -1;
    while (i<=j) {
        if (left) {
            int cmp;
            if (stanFirst) {
                cmp = -compare.cmp(stan, vals[i]);
            }else{
                cmp = compare.cmp(vals[i], stan);
            }
            if (cmp<=0) {
                if (cmp==0) equalIdx = i;
                i++;
            }else{
                left = false;
            }
        }else{
            int cmp;
            if (stanFirst) {
                cmp = -compare.cmp(stan, vals[j]);
            }else{
                cmp = compare.cmp(vals[j], stan);
            }
            if (cmp>0) {
                j--;
            }else{
                if (cmp==0) equalIdx = i;
                left = true;
                
                swap(vals[i], vals[j]);
                
                i++;
                j--;
            }
        }
    }
    
    swap(vals[i-1], vals[equalIdx]);
    
    return i-1;
}

void sortNutsAndBolts(vector<string> &nuts, vector<string> &bolts, Comparator &compare, int start, int end){
    if (start >= end) {
        return;
    }
    
    printf("[%d,%d]\n",start,end);
    int eq = NutsAndBoltsPartion(nuts, bolts[start], compare, start, end, false);
    NutsAndBoltsPartion(bolts, nuts[eq], compare, start, end, true);
    
    sortNutsAndBolts(nuts, bolts, compare, start, eq-1);
    sortNutsAndBolts(nuts, bolts, compare, eq+1, end);
}

void sortNutsAndBolts(vector<string> &nuts, vector<string> &bolts, Comparator compare) {
    sortNutsAndBolts(nuts, bolts, compare, 0, (int)nuts.size()-1);
}

int twoSum2(vector<int> &nums, int target) {
    sort(nums.begin(), nums.end());
    
    int result = 0, size = (int)nums.size();
    int i = 0, j = size-1;
    while (i<j) {
        if (nums[i]+nums[j]>target) {
            j--;
        }else{
            result += size-j-1;
            i++;
        }
    }
    
    while (i<size) {
        result += size-i-1;
        i++;
    }
    return result;
}

void wallsAndGates(vector<vector<int>> &rooms) {
    // write your code here
}



typedef enum {
    ///空地
    GridNodeTypeBlank,
    ///障碍物
    GridNodeTypeBarrier,
    ///出口
    GridNodeTypeExit,
}GridNodeType;

template <class T>
class GridIter{
public:
    typedef GridNodeType (*VisitPointFunc)(vector<vector<T>> &grid, Point curP, int depth, void *context);
    VisitPointFunc visitFunc;
    GridIter(VisitPointFunc visitFunc):visitFunc(visitFunc){};
};

//这样F第三个参数可以接受“任意可以转换了GridIter<T>”
template<class T, class F>
static void BFSGrid(vector<vector<T>> &grid, Point start, F visitFunc, void *context){
    BFSGrid(grid, start, (GridIter<T>)visitFunc, context);
}

template<class T>
static void BFSGrid(vector<vector<T>> &grid, Point &start, GridIter<T> iter, void *context){
    
    //最后一个位存放是否访问过，其他位存储深度
    int height = (int)grid.size(), width = (int)grid.front().size();
    int mark[height][width];
    memset(mark, 0, sizeof(mark));
    
    queue<Point> pointsQ;
    pointsQ.push(start);
    mark[start.x][start.y] = 1;
    GridNodeType type = iter.visitFunc(grid, start, 0, context);
    if (type==GridNodeTypeExit) {
        return;
    };
    
    while (!pointsQ.empty()) {
        auto p = pointsQ.front();
        pointsQ.pop();
        
        int depth = mark[p.x][p.y]>>1;
        Point round[4] = {{p.x+1, p.y},{p.x-1, p.y},{p.x, p.y+1},{p.x, p.y-1}};
        for (int i = 0; i<4; i++) {
            auto r = round[i];
            if (r.x<height && r.x>=0 && r.y<width && r.y>=0 && !(mark[r.x][r.y]&1)) {
                GridNodeType type = iter.visitFunc(grid, r, depth+1, context);
                mark[r.x][r.y] = ((depth+1)<<1)|1;
                if (type == GridNodeTypeExit){
                    return;
                }else{
                    pointsQ.push(r);
                }
            }
        }
    }
}

GridNodeType houseGridVisit(vector<vector<int>> &grid, Point curP, int depth, void *context){
    printf("[%d,%d] ",curP.x,curP.y);
    int *disSum = (int*)context;
    int width = (int)grid.front().size();
    
    if (grid[curP.x][curP.y] == 1) { //房子
        return GridNodeTypeBarrier;
    }else{
        disSum[curP.x*width+curP.y] += depth;
        return GridNodeTypeBlank;
    }
}

void countDis(int *counts, int size, int *dis){
    int leftC = counts[0], rightC = 0;
    for (int i = 1; i<size; i++) {
        dis[0] += counts[i]*i;
        rightC += counts[i];
    }
    
    for (int i = 1; i<size; i++) {
        dis[i] = dis[i-1]+(leftC-rightC);
        leftC += counts[i];
        rightC -= counts[i];
    }
}

/*
 1. BFS
 2. 暴力解，和1一样，复杂度为O(K*N^2)，K为房子的总数，房子密集时恐怖
 3. 分别求x和y的轴的距离，每个轴的求解，可以建一个一维数组，统计当前行或列上的房子个数，以这个一维数组就可以求解
 4. 同上，但求解单轴时，使用递推方法，上面时间复杂度O(n^2),这里是O(N),但遍历需要O(N^2)
 5. 求中卫数，对单轴而言，最近的距离就是中位数，但中位数位置可能有房子，所以还需要从中心向外扩散，复杂度为O(N)*k,k看房子是否集中。
 
 如果要考虑障碍物，则需要用到BFS，因为仅靠终点和起点没法计算距离，路径是绕的
 如果是单轴，就可以直接用中位数求法，这题是维度扩大后的变型
 */
int shortestDistance(vector<vector<int>> &grid) {
    int height = (int)grid.size(), width = (int)grid.front().size();
    
    int hc[width], vc[height];
    memset(hc, 0, sizeof(hc));
    memset(vc, 0, sizeof(vc));
    
    for (int i = 0; i<height; i++) {
        for (int j = 0; j<width; j++) {
            if (grid[i][j] == 1) {
                hc[j]++;
                vc[i]++;
            }
        }
    }
    
    printArrayOneLine(hc, width);
    printArrayOneLine(vc, height);
    
    int htc[width], vtc[height];
    memset(htc, 0, sizeof(hc));
    memset(vtc, 0, sizeof(vc));
    countDis(hc, width, htc);
    countDis(vc, height, vtc);
    
    printArrayOneLine(htc, width);
    printArrayOneLine(vtc, height);
    
    int minDis = INT_MAX;
    for (int i = 0; i<height; i++) {
        for (int j = 0; j<width; j++) {
            if (grid[i][j] == 0) {
                minDis = min(minDis, htc[j]+vtc[i]);
            }
        }
    }
    
    return minDis;
}

int maximumGap(vector<int> &nums) {
    if (nums.size()<2) {
        return 0;
    }
    
    int minVal = nums[0], maxVal = nums[0];
    for (int i = 1; i<nums.size(); i++) {
        minVal = min(nums[i], minVal);
        maxVal = max(nums[i], maxVal);
    }
    
    int averageGap = max(1, (maxVal-minVal)/(int)nums.size());
    int offset = minVal/averageGap;  //从最小值的桶开始计算，之前的桶不用了
    int length = maxVal/averageGap-offset+1;  //桶的个数
    pair<int, int> buckets[length]; //存储每个桶的最小值和最大值
    for (auto &p : buckets){
        p.first = -1;
        p.second = -1;
    }
    for (int i = 0; i<nums.size(); i++) {
        int bucketIdx = nums[i]/averageGap-offset;
        if (buckets[bucketIdx].first<0 || nums[i]<buckets[bucketIdx].first) {
            buckets[bucketIdx].first = nums[i];
        }
        if (buckets[bucketIdx].second<0 || nums[i]>buckets[bucketIdx].second) {
            buckets[bucketIdx].second = nums[i];
        }
    }
    
    int maxGap = 0, lastNum = buckets[0].second;
    for (int i = 1; i<length; i++) {
        if (buckets[i].first>0) {
            maxGap = max(maxGap, buckets[i].first-lastNum);
        }
        if (buckets[i].second>0) {
            lastNum = buckets[i].second;
        }
    }
    
    return maxGap;
}

int twoSumClosest(vector<int> &nums, int target) {
    sort(nums.begin(), nums.end());
    
    int i = 0, j = (int)nums.size()-1;
    
    int diff = INT_MAX;
    
    while (i<j) {
        int curDiff = nums[i]+nums[j]-target;
        
        if (curDiff>0) {
            diff = min(diff, curDiff);
            j--;  //换一个右侧点
        }else{
            diff = min(diff, -curDiff);
            i++;
        }
    }
    
    return diff;
}

int twoSum5(vector<int> &nums, int target) {
    sort(nums.begin(), nums.end());
    int i = 0, j = (int)nums.size()-1;
    int count = 0;
    
    while (i<j) {
        int curDiff = nums[i]+nums[j]-target;
        
        if (curDiff>0) {
            j--;  //换一个右侧点
        }else{
            count += j-i;
            i++;
        }
    }
    
    return count;
}

int twoSum6(vector<int> &nums, int target) {
    unordered_map<int, int> exists;
    float mid = target/2.0f;
    for (auto &i : nums){
        exists[i]++;
    }
    
    int count = 0;
    for (auto &p : exists){
        if (p.first<mid && exists.find(target-p.first) != exists.end()) {
            count++;
        }
    }
    
    if (!(target&1) && exists[mid]>1) {
        count++;
    }
    
    return count;
}

class TwoSum {
    unordered_map<int, int> counts;
public:
    void add(int number) {
        counts[number]++;
    }
    bool find(int value) {
        float mid = value/2.0f;
        for (auto &p : counts){
            if (p.first<mid && counts[value-p.first]>0) {
                return true;
            }
        }
        
        return !(value&1) && counts[mid]>1;
    }
};

vector<int> twoSum7(vector<int> &nums, int target) {
    unordered_map<int, int> exist;
    for (int i = 0; i<nums.size(); i++) {
        exist[nums[i]] = i;
    }
    for (int i = 0; i<nums.size(); i++) {
        int other = nums[i]-target;
        if (exist.find(other) != exist.end()) {
            return {min(exist[other], i)+1, max(exist[other], i)+1};
        }
    }
    return {};
}

vector<int> twoSum(vector<int> &nums, int target) {
    int i = 0, j = (int)nums.size()-1;
    while (i<j) {
        int sum = nums[i]+nums[j];
        if (target == sum) {
            return {i+1,j+1};
        }else if (target > sum){
            i++;
        }else{
            j--;
        }
    }
    return {};
}

bool treeNodeExist(TreeNode *root, int target){
    stack<TreeNode *> path;
    path.push(root);
    
    while (!path.empty()) {
        auto cur = path.top();
        if (cur->val==target) {
            return true;
        }
        
        path.pop();
        
        if (cur->right) {
            path.push(cur->right);
        }
        if (cur->left) {
            path.push(cur->left);
        }
    }
    
    return false;
}

bool wordPairCmp(pair<string, int>& wp1, pair<string, int> &wp2){
    int result = (int)wp1.first.length()-(int)wp2.first.length();
    if (result!=0) return result<0;
    result = wp1.first.back()-wp2.first.back();
    if (result!=0) return result<0;
    
    return wp1.first.compare(wp2.first)<0;
}

inline int prefixLapCount(string &s1, string &s2){
    int c = 0;
    while (s1[c] == s2[c]) {
        c++;
    }
    return c;
}

inline void wordAbb(string &originalWord, int prefixCount){
    int len = (int)originalWord.length();
    int cut = len-prefixCount-1;
    if (cut<=1) {
        return;
    }
    
    int destLen = prefixCount+1+log10(cut)+1;
    int preIdx = len-cut-2;
    string abb(destLen,' ');
    
    for (int i = 0; i<=preIdx; i++) {
        abb[i] = originalWord[i];
    }
    abb[destLen-1] = originalWord.back();
    
    for (int i = destLen-2; i>preIdx; i--) {
        abb[i] = cut%10+'0';
        cut = cut/10;
    }
    
    originalWord = abb;
}

vector<string> wordsAbbreviation(vector<string> &dict) {
    
    vector<pair<string, int>> wordPairs;
    int i = 0;
    for (auto &w : dict){
        wordPairs.push_back({w,i});
        i++;
    }
    sort(wordPairs.begin(), wordPairs.end(), wordPairCmp);
    
    printVectorPair(wordPairs);
    printf("\n-------------------\n");
    
    int size = (int)wordPairs.size();
    int preLapCount = 0; //和前一个重叠的字符个数
    for (int i = 0; i<size; i++) {
        
        int nextLapCount = 0;
        if (i<size-1 &&
            wordPairs[i].first.length() == wordPairs[i+1].first.length() &&
            wordPairs[i].first.back() == wordPairs[i+1].first.back()) {
            nextLapCount = prefixLapCount(wordPairs[i].first, wordPairs[i+1].first);
        }
        
        wordAbb(wordPairs[i].first, max(preLapCount, nextLapCount)+1);
        preLapCount = nextLapCount;
    }
    
    for (auto &p : wordPairs){
        dict[p.second] = p.first;
    }
    
    return dict;
}

vector<int> twoSum(TreeNode * root, int n) {
    if (root == nullptr) {
        return {};
    }
    float mid = n/2.0f;
    
    stack<TreeNode *> path;
    path.push(root);
    
    while (!path.empty()) {
        auto cur = path.top();
        if (cur->val<mid && treeNodeExist(root, n-cur->val)) {
            return {cur->val, n-cur->val};
        }
        
        path.pop();
        
        if (cur->right) {
            path.push(cur->right);
        }
        if (cur->left) {
            path.push(cur->left);
        }
    }
    
    return {};
}

int secondMax(vector<int> &nums) {
    int max1 = INT_MIN, max2 = INT_MIN;
    for (auto n : nums){
        if (n>max1) {
            max2 = max1;
            max1 = n;
        }else if (n>max2){
            max2 = n;
        }
    }
    
    return max2;
}

static bool multiSortCmp(vector<int> &v1, vector<int> &v2){
    int result = v1[1]-v2[1];
    if (result != 0) return result>0;
    return v1[0]<v2[0];
}

vector<vector<int>> multiSort(vector<vector<int>> &array) {
    sort(array.begin(), array.end(), multiSortCmp);
    return array;
}

template<class T>
class FlipTool{
public:
    static void flip(vector<T> &arr, int endIdx){
        int i = 0, j = endIdx;
        while (i<j) {
            swap(arr[i], arr[j]);
            i++;
            j--;
        }
    }
};

void pancakeSort(vector<int> &array) {
    int size = (int)array.size(), k = size;
    while (k>1) {
        int maxVal = INT_MIN, maxIdx = 0;
        for (int i = 0; i<k; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        
        FlipTool<int>::flip(array, maxIdx);
        FlipTool<int>::flip(array, k-1);
        k--;
    }
}

int countMinDis(int *counts, int size){
    int leftC = counts[0], rightC = 0;
    int curDis = 0;
    for (int i = 1; i<size; i++) {
        curDis += counts[i]*i;
        rightC += counts[i];
    }
    
    int lastDis = curDis;
    for (int i = 1; i<size; i++) {
        curDis += leftC-rightC;
        if (leftC>rightC) {
            curDis=min(lastDis, curDis);
            break;
        }
        leftC += counts[i];
        rightC -= counts[i];
        lastDis = curDis;
    }
    return curDis;
}

int minTotalDistance(vector<vector<int>> &grid) {
    int height = (int)grid.size(), width = (int)grid.front().size();
    
    int hc[width], vc[height];
    memset(hc, 0, sizeof(hc));
    memset(vc, 0, sizeof(vc));
    
    for (int i = 0; i<height; i++) {
        for (int j = 0; j<width; j++) {
            if (grid[i][j] == 1) {
                hc[j]++;
                vc[i]++;
            }
        }
    }
    return countMinDis(hc, width)+countMinDis(vc, height);
}

int minMeetingRooms(vector<Interval> &intervals) {
    vector<pair<int, bool>> timeP;
    for (auto &i : intervals){
        timeP.push_back({i.start, true});
        timeP.push_back({i.end, false});
    }
    
    sort(timeP.begin(), timeP.end(), PairSort<int, bool>::pairFirstComp);
    int maxCount = INT_MIN, count = 0;
    for (auto &t : timeP){
        count += t.second?1:-1;
        maxCount = max(maxCount, count);
    }
    
    return maxCount;
}

void showCharCounts(int *chars){
    for (int i = 0; i<256; i++) {
        if (chars[i]>0) {
            printf("%c %d, ",i,chars[i]);
        }
    }
    printf("\n\n");
}

int majorityNumber(vector<int> &nums, int k) {
    unordered_map<int, int> counts;
    for (auto &n : nums){
        counts[n]++;
    }
    
    int times = (int)nums.size()/k;
    for (auto &p : counts){
        if (p.second>times) {
            return p.first;
        }
    }
    
    return 0;
}



//这也属于双指针的问题
string minWindow(string &source , string &target) {
    if (source.empty() || target.empty()) { //错误8，没做空值判断
        return "";
    }
    int charR = 256;
    int charC1[charR], charC2[charR];
    memset(charC1, 0, sizeof(charC1));
    memset(charC2, 0, sizeof(charC2));
    for (auto &c : target){
        charC1[c]++;
    }
    
    int i = 0, j = 0, size = (int)source.size();
    bool involved = true;
    do {
        charC2[source[j]]++;
        involved = true;
        for (int k = 0; k<charR; k++) {
            if (charC1[k]>charC2[k]) {
                involved = false;
                break;
            }
        }
        //错误1，没及时退出，j++执行
        if (involved) break;
        j++;
    } while (j<size);
    
    if (!involved) {
        return "";
    }
    
    //i推到一个位置：减去这个i上的字符，就会不满足
    while (i<=j){ //错误2，没做限制
        if ((--charC2[source[i]])<charC1[source[i]]) {//错误3,&&后面可能不执行
            charC2[source[i]]++;
            break;
        }else{
            i++;
        }
    }
    
    int minLen = j-i+1, minStart = i;
    while (1) {
        //        cout<<source.substr(i, j-i+1)<<endl;
        //        showCharCounts(charC2);
        //减掉尾部的一个必要元素
        char discard = 0;  //错误4，i位置的元素不一定是唯一的，以前是唯一，但后面j往前推，可能会加入重复的进来，所以要做两次必要元素判断
        while (i<=j){
            if ((--charC2[source[i]])<charC1[source[i]]) {
                if (!discard) {
                    discard = source[i];
                    i++;
                }else{
                    charC2[source[i]]++;
                    break;
                }
            }else{
                i++;
            }
        }
        
        //        cout<<source.substr(i, j-i+1)<<endl;
        //        showCharCounts(charC2);
        
        //再在头部加入这个必要元素
        do {
            j++;
            charC2[source[j]]++; //错误5，没有更新charC2
        } while (j<size && source[j]!=discard);//错误6，没有对j做控制
        
        if (j == size) { //错误7，退出要在计算最小高度之前，因为最后一次可能不是满足条件的
            break;
        }
        
        if (j-i+1<minLen) {
            minLen = j-i+1;
            minStart = i;
        }
        
    }
    
    return source.substr(minStart, minLen);
}

vector<string> nameDeduplication(vector<string> &names) {
    
    unordered_map<string, bool> exist;
    vector<string> result;
    for (auto &n : names){
        transform(n.begin(),n.end(),n.begin(),::tolower);
        if (exist.find(n) == exist.end()) {
            exist[n] = true;
            result.push_back(n);
        }
    }
    
    return result;
}

class LoadBalancer {
    unordered_set<int> servers;
public:
    LoadBalancer() {
        
    }
    void add(int server_id) {
        servers.insert(server_id);
    }
    
    void remove(int server_id) {
        servers.erase(server_id);
    }
    
    int pick() {
        //        srand((int)time(0));
        int idx = random()%(int)servers.size();
        auto iter = servers.begin();
        while (idx>0) {
            iter++;
            idx--;
        }
        return *iter;
    }
};

class Memcache {
    struct cacheData{
        int val;
        int deadTime; //最后的有效时间,为0表示永远有效
    };
    
    unordered_map<int,cacheData> caches;
    static const int unexistVal = INT_MAX;
    cacheData *getCache(int curtTime, int key){
        auto c = caches.find(key);
        if (c == caches.end()) {
            return nullptr;
        }
        if (c->second.deadTime>0 && c->second.deadTime<curtTime) {
            caches.erase(c);
            return nullptr;
        }
        return &(c->second);
    }
public:
    Memcache() {
        
    }
    int get(int curtTime, int key) {
        auto cache = getCache(curtTime, key);
        if (cache) {
            return cache->val;
        }else{
            return unexistVal;
        }
    }
    void set(int curtTime, int key, int value, int ttl) {
        if (ttl) {
            caches[key] = {value, curtTime+ttl-1};
        }else{
            caches[key] = {value, 0};
        }
    }
    
    void del(int curtTime, int key) {
        auto c = caches.find(key);
        if (c == caches.end()) {
            return;
        }
        caches.erase(c);
    }
    int incr(int curtTime, int key, int delta) {
        auto cache = getCache(curtTime, key);
        if (cache == nullptr) {
            return unexistVal;
        }
        return cache->val += delta;
    }
    int decr(int curtTime, int key, int delta) {
        auto cache = getCache(curtTime, key);
        if (cache == nullptr) {
            return unexistVal;
        }
        return cache->val -= delta;
    }
};

class CountingBloomFilter {
    unordered_map<string, int> exist;
public:
    
    CountingBloomFilter(int k) {
        
    }
    
    void add(string &word) {
        exist[word]++;
    }
    void remove(string &word) {
        auto i = exist.find(word);
        if (i == exist.end()) {
            return;
        }
        if (--(i->second)==0) {
            exist.erase(i);
        }
    }
    bool contains(string &word) {
        return exist.find(word) != exist.end();
    }
};

class StandardBloomFilter {
    static const int maxSize = (1<<23)-1;
    
    vector<TFDataStruct::StringHash> hashGroup;
    bitset<maxSize> marks;
public:
    
    StandardBloomFilter(int k) {
        srand(49827231);
        for (int i = 0; i<k; i++) {
            long long p = (1<<(random()%23))-1; //小于mod
            hashGroup.push_back(TFDataStruct::StringHash(p, maxSize));
        }
    }
    
    void add(string &word) {
        for (auto &h : hashGroup){
            long long pos = h.hash(word);
            printf("%lld",pos);
            marks[pos] = true;
        }
    }
    
    bool contains(string &word) {
        for (auto &h : hashGroup){
            long long pos = h.hash(word);
            printf("%lld",pos);
            if (!marks[pos]) {
                return false;
            }
        }
        
        return true;
    }
};

bool isStrobogrammatic(string &num) {
    //为0的都是没有镜像对的数字
    unsigned char pair[10] = {'0','1',0,0,0,0,'9',0,'8','6'};
    
    int i = 0, j = (int)num.size()-1;
    while (i<=j) {
        if (pair[num[i]-'0']==num[j]) {
            i++;
            j--;
        }else{
            return false;
        }
    }
    
    return true;
}

vector<int> findAnagrams(string &s, string &p) {
    int lens = (int)s.length(),lenp = (int)p.length() ;
    if (lens<lenp) {
        return {};
    }
    
    int count1[26];
    memset(count1, 0, sizeof(count1));
    for (auto &c : p){
        count1[c-'a']++;
    }
    
    
    int count2[26];
    memset(count2, 0, sizeof(count2));
    for (int i = 0; i<lenp; i++){
        count2[s[i]-'a']++;
    }
    
    int charDiff[26];
    int diffCount = 0;
    for (int i = 0; i<26; i++) {
        charDiff[i] = count2[i]-count1[i];
        if (charDiff[i] != 0) {
            diffCount++;
        }
    }
    
    int leftIdx = 0;
    vector<int> result;
    do {
        if (diffCount==0) {
            result.push_back(leftIdx);
        }
        if (leftIdx==lens-lenp) {
            break;
        }
        
        int add = s[leftIdx+lenp]-'a';
        charDiff[add]++;
        if (charDiff[add]==0) {
            diffCount--;
        }else if (charDiff[add]==1){
            diffCount++;
        }
        
        int discard = s[leftIdx]-'a';
        charDiff[discard]--;
        if (charDiff[discard]==0) {
            diffCount--;
        }else if (charDiff[discard]==-1){
            diffCount++;
        }
        leftIdx++;
        
        //        for (int i = 0; i<26; i++) {
        //            if (charDiff[i]!=0) {
        //                printf("%c %d, ",i+'a',charDiff[i]);
        //            }
        //        }
        //        printf(" diff %d\n",diffCount);
        
    } while (1);
    
    return result;
}

inline void wordAbb(string &originalWord, int prefixCount){
    int len = (int)originalWord.length();
    int cut = len-prefixCount-1;
    if (cut<1) {
        return;
    }
    
    int destLen = prefixCount+1+log10(cut)+1;
    int preIdx = len-cut-2;
    string abb(destLen,' ');
    
    for (int i = 0; i<=preIdx; i++) {
        abb[i] = originalWord[i];
    }
    abb[destLen-1] = originalWord.back();
    
    for (int i = destLen-2; i>preIdx; i--) {
        abb[i] = cut%10+'0';
        cut = cut/10;
    }
    
    originalWord = abb;
}

class ValidWordAbbr {
    unordered_map<string, string> abbs;
public:
    ValidWordAbbr(vector<string> dictionary) {
        for (auto &s : dictionary){
            if (s.length()<3) {
                continue;
            }
            auto sc = s;
            wordAbb(sc, 1);
            auto i = abbs.find(sc);
            //有一个单词有这个缩写的，记录这个单词；
            //多个单词有这个缩写的，必定重复，所以只做标记，存入空字符串
            if (i == abbs.end()) {
                abbs[sc] = s;
            }else{
                if (i->second.compare(s) != 0) {
                    abbs[sc] = "";
                }
            }
        }
    }
    bool isUnique(string &word) {
        if (word.length()<3) {
            return true;
        }
        auto wc = word;
        wordAbb(wc, 1);
        auto i = abbs.find(wc);
        if (i==abbs.end()) {
            return true;
        }
        
        //空字符串时，有多个单词共缩写，必定重复，跟这个式子是匹配的
        return i->second.compare(word)==0;
    }
};

struct SparceMatrixNode{
    int i,j;
    int v;
};

inline void convertToSparceMatrix(vector<SparceMatrixNode> &nodes, vector<vector<int>> &matrix){
    int i = 0;
    for (auto &v:matrix){
        int j = 0;
        for (auto &n:v){
            if (n!=0) {
                nodes.push_back({i,j,n});
            }
            j++;
        }
        i++;
    }
}

vector<vector<int>> multiply(vector<vector<int>> &A, vector<vector<int>> &B) {
    vector<SparceMatrixNode> nodesA;
    vector<SparceMatrixNode> nodesB;
    
    convertToSparceMatrix(nodesA, A);
    convertToSparceMatrix(nodesB, B);
    
    vector<vector<int>> result(A.size(), vector<int>(B.front().size(), 0));
    for (auto &na : nodesA){
        for (auto &nb : nodesB){
            if (na.j == nb.i) {
                result[na.i][nb.j] += na.v*nb.v;
            }
        }
    }
    
    return result;
}

class RandomizedSet {
    //值和值的索引
    unordered_map<int,int> valIndexes;
    vector<int> vals = vector<int>(1<<20,0);
    int size = 0;
public:
    RandomizedSet() {
        srand((uint)time(0));
    }
    
    bool insert(int val) {
        if (valIndexes.find(val) != valIndexes.end()) {
            return false;
        }
        valIndexes[val]=size;
        //        vals.push_back(val);
        vals[size] = val;
        size++;
        return true;
    }
    
    bool remove(int val) {
        auto i = valIndexes.find(val);
        if (i == valIndexes.end()) {
            return false;
        }
        
        //调换被删除元素和最后一个元素的位置，这样之间的元素的索引不变
        valIndexes[vals[size-1]] = i->second;
        vals[i->second] = vals[size-1];
        size--;
        
        valIndexes.erase(i);
        return true;
    }
    
    int getRandom() {
        //有了数组，便可以做随机访问了
        int idx = rand()%size;
        //        printf("%d\n",idx);
        return vals[idx];
    }
};

vector<string> missingString(string &str1, string &str2) {
    str1.push_back(' ');
    str2.push_back(' ');
    unordered_map<string, vector<int>> wordsIdx1;
    int start = -1, idx = 0;
    int wordCount = 0;
    while (idx<str1.length()) {
        if (start<0) {
            if (str1[idx] != ' ') {
                start = idx;
            }
        }else{
            if (str1[idx]==' ') {
                auto word = str1.substr(start, idx-start);
                if (wordsIdx1.find(word) == wordsIdx1.end()) {
                    wordsIdx1[word] = {wordCount};
                }else{
                    wordsIdx1[word].push_back(wordCount);
                }
                wordCount++;
                start = -1;
            }
        }
        idx++;
    }
    
    start = -1;
    idx = 0;
    while (idx<str2.length()) {
        if (start<0) {
            if (str2[idx] != ' ') {
                start = idx;
            }
        }else{
            if (str2[idx]==' ') {
                auto iter = wordsIdx1.find(str2.substr(start, idx-start));
                start = -1;
                if (iter != wordsIdx1.end()) {
                    wordsIdx1.erase(iter);
                }
            }
        }
        idx++;
    }
    
    vector<pair<string, int>> pairs;
    for (auto &p : wordsIdx1){
        for (auto &i : p.second){
            pairs.push_back({p.first, i});
        }
    }
    sort(pairs.begin(), pairs.end(), PairSort<string, int>::pairSecondComp);
    
    vector<string> result;
    for (auto &p : pairs){
        result.push_back(p.first);
    }
    return result;
}

int FindElements(vector<vector<int>> &Matrix) {
    int rowCount = (int)Matrix.size();
    int charSize = rowCount/64;
    unordered_map<int, uint64_t*> marks;
    int i = 0;
    for (auto &v : Matrix){
        for (auto &n:v){
            auto iter = marks.find(n);
            if (iter == marks.end()) {
                uint64_t *newMark = new uint64_t[charSize+1];
                memset(newMark, 0, sizeof(uint64_t)*(charSize+1));
                marks[n] = newMark;
                
                iter = marks.find(n);
            }
            int charIdx = i/64, innerIdx = i%64;
            iter->second[charIdx] |= (1ull<<(63-innerIdx)); //每行，出现了某个数，只会标记一次，多次标记会覆盖
            if (n==4) {
                printf("%llu \n",iter->second[charIdx]);
            }
        }
        i++;
    }
    
    uint64_t suffix = -(1ull<<(64-rowCount%64));
    for (auto &iter : marks){
        
        bool error = false;
        for (int k = 0; k<charSize; k++) {
            if (iter.second[k] != -1ll) {
                error = true;
                break;
            }
        }
        if (!error && iter.second[charSize]==suffix) {
            return iter.first;
        }
    }
    return 0;
}

vector<string> findRepeatedDna(string &s) {
    if (s.length()<10){
        return {};
    }
    unordered_map<string, short> exist;
    int i = 0;
    vector<string> result;
    do {
        auto &count = exist[s.substr(i, 10)];
        if (++count==2) {
            result.push_back(s.substr(i, 10));
        }
        i++;
    } while (i<s.length()-9);
    
    return result;
}

bool wordPattern(string &pattern, string &teststr) {
    teststr.push_back(' ');
    
    //1. 前缀树记录的是单词到字符的对应关系，每个节点唯一对应一个单词(即从跟到这个节点的路径拼起来的单词),所以用节点存储单词映射的字符；
    //2. markC2W[c]表示c这个字符映射的单词，又因为节点和单词的一一对应关系，所以把单词对应的节点存入
    TFDataStruct::Trie<char> trieW2C;
    TFDataStruct::Trie<char>::TrieNode* charNodes[26];
    memset(charNodes, 0, sizeof(charNodes));
    
    //3. 因为要一一对应的关系，1和2刚好是两个方向
    
    int start=-1, i = 0;
    int wordIdx = 0;
    for (auto &c : teststr){
        
        if (start<0) {
            if (c!=' ') {
                start = i;
            }
        }else{
            if (c==' ') {
                auto str = teststr.substr(start, i-start);
                auto node = trieW2C.insert(str);
                
                auto charNode = charNodes[pattern[wordIdx]-'a'];
                if (charNode == nullptr) {
                    if (node->relateData>0) {
                        return false;
                    }
                }else if (node != charNode) {
                    return false;
                }
                
                node->relateData = pattern[wordIdx];
                charNodes[pattern[wordIdx]-'a'] = node;
                
                start = -1;
                wordIdx++;
            }
        }
        
        i++;
    }
    
    return true;
}

void maxPalindromeLen(string &str, int*oddLens, int *evenLens){
    int size = (int)str.length();
    oddLens[0] = 1;
    
    //奇数长度回文求一遍
    int maxI=0, maxR=1; //maxR是已求回文部分，最右侧的边界，maxI是这个回文的中心
    for (int i = 1; i<size; i++) {
        int curLen = 1; //这个不是回文的总长度，而是一半，准确的说是左边或者右边的长度，即包含中心
        if (i<maxR) {
            int mirrorLen = oddLens[2*maxI-i];
            if (mirrorLen+i<maxR) {//i...mirrorLen+i-1,[至少一个间隔],maxR
                curLen = mirrorLen;
            }else{
                curLen = maxR-i;
                int j = maxR;
                while (j<size && (2*i-j)>=0 && str[j]==str[2*i-j]) {
                    j++;
                    curLen++;
                }
                if (j>maxR) {  //能到达更远的地方，更新
                    maxR = j;
                    maxI = i;
                }
            }
        }else{
            int j = i+1;
            while (j<size && (2*i-j)>=0 && str[j]==str[2*i-j]) {
                j++;
                curLen++;
            }
            if (j>maxR) {  //能到达更远的地方，更新
                maxR = j;
                maxI = i;
            }
        }
        
        oddLens[i] = curLen;
    }
    
    //偶数回文求一遍
    maxI=-1;
    maxR=0;
    for (int i = 0; i<size; i++) {
        int curLen = 0; //偶数时，是准确的一半，中心偏左，这里的长度不包含中心
        if (i<maxR) {
            int mirrorLen = 2*maxI-i>=0?evenLens[2*maxI-i]:0;
            if (mirrorLen+i<maxR-1) {//i...mirrorLen+i,[至少一个间隔],maxR
                curLen = mirrorLen;
            }else{
                curLen = maxR-i-1;
                int j = maxR;
                while (j<size && (2*i+1-j)>=0 && str[j]==str[2*i+1-j]) {
                    j++;
                    curLen++;
                }
                if (j>maxR) {  //能到达更远的地方，更新
                    maxR = j;
                    maxI = i;
                }
            }
        }else{
            int j = i+1;
            while (j<size && (2*i+1-j)>=0 && str[j]==str[2*i+1-j]) {
                j++;
                curLen++;
            }
            if (j>maxR) {  //能到达更远的地方，更新
                maxR = j;
                maxI = i;
            }
        }
        
        evenLens[i] = curLen;
    }
    
    for (int i = 0; i<size; i++) {
        oddLens[i] = oddLens[i]*2-1;
        evenLens[i] = evenLens[i]*2;
    }
}

void maxPalindromeLen(string &str, int*lens){
    int size = (int)str.length();
    
    //这样处理，每个位置的回文长度都是奇数，1221=>1#2#2#1,第一个2位置的偶数回文变成了中心#的奇数回文
    //并且处理后左侧的长度就是原回文总长度，也就是处理后总长/2==原总长；所以len存储总长/2。
    int oddLen = 2*size+1;
    string oddStr(oddLen, '#');
    for (int i = 0; i<size; i++) {
        oddStr[2*i+1]=str[i];
    }
    
    int oddLens[oddLen];
    memset(oddLens, 0, sizeof(oddLens));
    oddLens[0] = 1;
    
    int maxI=0, maxR=1; //maxR是已求回文部分，最右侧的边界，maxI是这个回文的中心
    for (int i = 1; i<oddLen; i++) {
        int curLen = 1; //这个不是回文的总长度，而是一半，准确的说是左边或者右边的长度，即包含中心
        if (i<maxR) {
            
            int mirrorLen = oddLens[2*maxI-i];
            if (mirrorLen+i<maxR) {//i...mirrorLen+i-1,[至少一个间隔],maxR
                curLen = mirrorLen;
            }else{
                curLen = maxR-i;
                int j = maxR;
                while (j<oddLen && (2*i-j)>=0 && oddStr[j]==oddStr[2*i-j]) {
                    j++;
                    curLen++;
                }
                if (oddStr[j-1]=='#'){
                    j--;
                    curLen--;
                }
                if (j>maxR) {  //能到达更远的地方，更新
                    maxR = j;
                    maxI = i;
                }
            }
        }else{
            int j = i+1;
            while (j<oddLen && oddStr[j]==oddStr[2*i-j]) {
                j++;
                curLen++;
            }
            if (oddStr[j-1]=='#'){
                j--;
                curLen--;
            }
            if (j>maxR) {  //能到达更远的地方，更新
                maxR = j;
                maxI = i;
            }
        }
        
        printf("%d ",curLen);
        oddLens[i] = curLen;
    }
    
    for (int i = 0; i<size; i++) {
        lens[i] = max(oddLens[2*i+1], oddLens[2*i+2]);
    }
    
    printf("\n");
}

inline string reverseSub(string &str, int len, int end){
    string sub(len, ' ');
    for (int i = 0; i<len; i++) {
        sub[i] = str[end-i];
    }
    return sub;
}

vector<vector<int>> palindromePairs(vector<string> &words) {
    unordered_map<string, short> exists;
    int idx = 0;
    for (auto &w:words){
        exists[w]=idx;
        idx++;
    }
    
    vector<vector<int>> result;
    
    idx = 0;
    for (auto &w:words){
        if (w.empty()) {
            continue;
        }
        int wLen = (int)w.length();
        int oddLens[wLen];
        int evenLens[wLen];
        maxPalindromeLen(w, oddLens, evenLens);
        
        //        printArrayOneLine(oddLens, wLen);
        //        printArrayOneLine(evenLens, wLen);
        
        for (int i = 0; i<wLen; i++) {
            int pLens[2] = {oddLens[i], evenLens[i]};
            for (int j = 0; j<2; j++) {
                int pLen = pLens[j];
                if (pLen == 0) continue;
                if ((pLen-1)/2==i) {
                    auto iter = exists.find(reverseSub(w, wLen-pLen, wLen-1));
                    if (iter!=exists.end()) {
                        result.push_back({iter->second, idx});
                        if (pLen==wLen) { //自身是回文，对方是空字符串的特殊情况，空字符串还可接左边
                            result.push_back({idx, iter->second});
                        }
                    }
                }else if (pLen/2+i==wLen-1){
                    auto iter = exists.find(reverseSub(w, wLen-pLen, wLen-pLen-1));
                    if (iter!=exists.end()) {
                        result.push_back({idx, iter->second});
                    }
                }
            }
        }
        
        //自身全部逆转,如果存在，则接左边或右边都是回文,为了不重复计算，只记接右边的
        auto totalIter = exists.find(reverseSub(w, wLen, wLen-1));
        if (totalIter != exists.end() && totalIter->second != idx) {
            result.push_back({idx, totalIter->second});
        }
        
        idx++;
    }
    
    return result;
}

string longestPalindrome(string &s) {
    if (s.empty()) {
        return "";
    }
    int oddLens[s.length()];
    int evenLens[s.length()];
    maxPalindromeLen(s, oddLens, evenLens);
    printArrayOneLine(oddLens, s.length());
    printArrayOneLine(evenLens, s.length());
    int maxIdx = 0, maxLen = INT_MIN;
    for (int i = 0; i<s.length(); i++) {
        int ml = max(oddLens[i], evenLens[i]);
        if (ml>maxLen) {
            maxLen = ml;
            maxIdx = i;
        }
    }
    
    return s.substr(maxIdx-(maxLen-1)/2, maxLen);
}

struct IdxNode{
    uint8_t val;
    uint8_t next;
};

vector<int> anagramMappings(vector<int> &A, vector<int> &B) {
    uint8_t size = (uint8_t)A.size();
    unordered_map<int, vector<int>> idxB;
    
    for (uint8_t i = 0; i<size; i++) {
        idxB[B[i]].push_back(i);
    }
    
    vector<int> result;
    for (int i = 0; i<size; i++) {
        auto &idxes = idxB[A[i]];
        result.push_back(idxes.back());
        idxes.pop_back();
    }
    
    return result;
}

int subarraySumEqualsK(vector<int> &nums, int k) {
    int size = (int)nums.size();
    unordered_map<int, vector<int>> exist;
    
    int sum = 0;
    for (int i = 0; i<size; i++) {
        sum += nums[i];
        printf("%d ",sum);
        exist[sum].push_back(i);
    }
    printf("\n");
    
    auto i = exist.find(k);
    int result = i == exist.end()?0:i->second.size();
    for (auto &c : exist){
        auto iter = exist.find(c.first+k);
        if (iter != exist.end()) {
            for (auto &idx1 : c.second){
                for (auto &idx2 : iter->second){
                    if (idx2>idx1) {
                        printf("(%d, %d),%d-%d=%d\n",idx1,idx2,iter->first,c.first,k);
                    }
                    result += idx2>idx1?1:0;
                }
            }
        }
    }
    
    return result;
}

bool isSentenceSimilarity(vector<string> &words1, vector<string> &words2, vector<vector<string>> &pairs) {
    
    if (words1.size() != words2.size()) {
        return false;
    }
    
    unordered_map<string, bool> relations;
    
    for (auto &p : pairs) {
        relations[p.front()+p.back()] = true;
        relations[p.back()+p.front()] = true;
    }
    
    for (int i = 0; i<words1.size(); i++) {
        if (words1[i].compare(words2[i]) == 0) {
            continue;
        }
        if (relations.find(words1[i]+words2[i]) == relations.end()) {
            return false;
        }
    }
    
    return true;
}


/**
 求网格图的联通分量
 
 @param points 得到的值为1的点集合
 @param grid 网格
 @param start 起始点
 @param mark 点是否已访问的标记
 */
void gridConnectComp(vector<Point> &points, vector<vector<int>> &grid, Point start, bool *mark){
    int height = (int)grid.size(), width = (int)grid.front().size();
    stack<Point> path;
    path.push(start);
    
    while (!path.empty()) {
        auto cur = path.top();
        points.push_back(cur);
        path.pop();
        
        cur.y--;
        if (cur.y>=0 && grid[cur.y][cur.x]==1 && mark[width*cur.y+cur.x]==0) {
            mark[width*cur.y+cur.x] = 1;
            path.push(cur);
        }
        cur.y+=2;
        if (cur.y<height && grid[cur.y][cur.x]==1 && mark[width*cur.y+cur.x]==0) {
            mark[width*cur.y+cur.x] = 1;
            path.push(cur);
        }
        cur.y--;
        cur.x--;
        if (cur.x>=0 && grid[cur.y][cur.x]==1 && mark[width*cur.y+cur.x]==0) {
            mark[width*cur.y+cur.x] = 1;
            path.push(cur);
        }
        cur.x+=2;
        if (cur.x<width && grid[cur.y][cur.x]==1 && mark[width*cur.y+cur.x]==0) {
            mark[width*cur.y+cur.x] = 1;
            path.push(cur);
        }
    }
}

//只有平移后位置相同的岛才认为是相同的
int numberofDistinctIslands(vector<vector<int>> &grid) {
    int height = (int)grid.size(), width = (int)grid.front().size();
    bool mark[height][width];
    memset(mark, 0, sizeof(mark));
    
    unordered_set<string> islands;
    
    for (int i = 0; i<height; i++) {
        for (int j = 0; j<width; j++) {
            if (mark[i][j]==0 && grid[i][j]==1) {
                mark[i][j]=1;
                vector<Point> points;
                gridConnectComp(points, grid, {j,i}, (bool*)mark);
                
                vector<int> idxes;
                for (auto &p : points){
                    idxes.push_back(p.x*width+p.y);
                }
                sort(idxes.begin(), idxes.end());
                int minIdx = idxes.front();
                
                //变长不超过50，总长不会超过250，所以每个idx可以转换成ASCII码
                string key;
                for (auto &i : idxes){
                    key.push_back(i-minIdx); //减去最小值是为了把岛平移到原点，即最小值作原点，这样相同样子的岛会到达同一个位置
                }
                
                islands.insert(key);
            }
        }
    }
    
    return (int)islands.size();
}

static bool pointCmpXFirst(Point &p1, Point &p2){
    int ret = p1.x-p2.x;
    return ret!=0?ret<0:p1.y<p2.y;
}

inline string tagForPoints(vector<Point> &points, int width){
    
    sort(points.begin(), points.end(), pointCmpXFirst);
    string tag;
    for (auto &p:points){
        tag += to_string(p.x)+" "+to_string(p.y)+",";
    }
    cout<<tag<<endl;
    return tag;
}

inline bitset<250> tagForPoints2(vector<Point> &points, int width){
    bitset<250> tag;
    for(auto &p:points){
        tag[p.y*width+p.x] = 1;
    }
    return tag;
}

//把翻转和旋转后的各种类型的标记都存入set
void insertRotateFlipTags(vector<Point> &points, unordered_map<string, int> &tags, int width){
    int identifier = (int)tags.size();
    //整体操作是：旋转3次，翻转一次，再旋转3次，这样所有一体的8种形态都有了
    for (int i = 0; i<7; i++) {
        if (i==3) { //翻转
            for (auto &p:points){
                p.x = -p.x;
                p.y = p.y;
            }
        }else{
            for (auto &p:points){
                auto temp = p.x;
                p.x = -p.y;
                p.y = temp;
            }
        }
        
        //找到最小点
        Point firstP=points.front();
        for (auto &p:points){
            if (p.y<firstP.y) {
                firstP = p;
            }else if (p.y==firstP.y && p.x<firstP.x){
                firstP = p;
            }
        }
        
        //按最小点移动到原点，平移整个形状
        for (auto &p : points){
            p.x -= firstP.x;
            p.y -= firstP.y;
        }
        
        tags[tagForPoints(points, width)] = identifier;  //同一组的设置同一个id
    }
}

//岛可以平移、翻转或旋转来得到相同位置，认为是同一个岛
int numDistinctIslands2(vector<vector<int>> &grid) {
    int height = (int)grid.size(), width = (int)grid.front().size();
    bool mark[height][width];
    memset(mark, 0, sizeof(mark));
    
    unordered_map<string, int> tags;
    
    for (int i = 0; i<height; i++) {
        for (int j = 0; j<width; j++) {
            if (mark[i][j]==0 && grid[i][j]==1) {
                mark[i][j]=1;
                vector<Point> points;
                gridConnectComp(points, grid, {j,i}, (bool*)mark);
                
                for (auto &p : points){
                    p.x -= j;
                    p.y -= i;
                }
                
                auto tag = tagForPoints(points, width);
                if (tags.find(tag) == tags.end()) {
                    //直接tags[tag] = (int)tags.size()+1可能出错，如果编译后先求左边，size会先增加了1
                    int size = (int)tags.size()+1;
                    tags[tag] = size;
                    insertRotateFlipTags(points, tags, width);
                }
            }
        }
    }
    
    unordered_set<int> kinds;
    for (auto &t:tags){
        kinds.insert(t.second);
    }
    return (int)kinds.size();
}

static bool isReflectedCmp(vector<int> &p1, vector<int> &p2){
    int ret = p1[1]-p2[1];
    return ret!=0?ret<0:p1[0]<p2[0];
}

bool isReflected(vector<vector<int>> &points) {
    int size = (int)points.size();
    if (size<2) {
        return true;
    }
    sort(points.begin(), points.end(), isReflectedCmp);
    
    int i = 0, j = 0;
    int midX = INT_MIN;
    while (1) {
        //找到下一个y不同的地方，那么[i,j]区间内y都是相等的
        while (j<size-1 && points[j][1]==points[j+1][1]) {
            j++;
        }
        
        int curMidX = 0;
        int midIdx = i+(j-i)/2;
        if (((j-i)&1)==0) { //j-i+1是这一段的个数，为奇数时，有一个点没有匹配的点
            curMidX = points[midIdx][0];
        }else{
            curMidX = (points[midIdx][0]+points[midIdx+1][0])/2;
        }
        
        if (midX == INT_MIN) {
            midX = curMidX;
        }else if (midX != curMidX){
            return false;
        }
        
        int next = j+1;
        
        //i向右，j向左，不断检测相邻点的x差值，保证两边是一样的，这样才能对称
        while (i<j-1) {
            if (points[i][0]+points[j][0] == 2*curMidX) {
                i++;
                j--;
            }else{
                return false;
            }
        }
        
        if (next==size) {
            break;
        }
        i = j = next;
    }
    
    return true;
}

#define LRUCache(c) LRUCache cache(c);

#endif /* page2_h */
