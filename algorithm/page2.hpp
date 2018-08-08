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
    
    printf("\n");
    printVectorIntOneLine(uglyNums);
    
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


#endif /* page2_h */
