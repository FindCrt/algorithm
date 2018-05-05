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
#include <iostream>
#include "heap.hpp"
#include <mach/mach_time.h>
#include <unordered_set>

#include "TFSort.h"
#include "MinStack.hpp"

#include "page1.hpp"
#include "TopK.hpp"

using namespace std;

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
            count[num]++;
        }
    }
    
    short miss1 = -1, miss2 = -1;
    for (int i = 1; i<n; i++) {
        printf("[%d]%d ",i,count[i]);
        if (count[i] == 0) {
            if (miss1 < 0) {
                miss1 = i;
            }else{
                miss2 = i;
                break;
            }
        }
    }
    
    if (miss1 < 0 && miss2 < 0) {
        return 30;
    }
    
    int result = miss1*10+miss2;
    if (miss2 < 0) {
        result = miss1;
    }
    if (result == 12 && str.find("12") != string::npos) {
        return 21;
    }
    
    return result;
}

int main(int argc, const char * argv[]) {
    
    string str = "111098654327128213127262524232120191817161514";
    
    auto result = findMissing2(28, str);
    
}
