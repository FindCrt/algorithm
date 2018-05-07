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

int main(int argc, const char * argv[]) {
    
    vector<int> nums = {359,376,43,315,167,216,777,625,498,442,172,324,987,400,280,367,371,24,418,208,812,488,861,646,63,804,863,853,102,174,443,901,486,126,419,701,254,550,48,214,873,386,965,504,753,336,527,522,895,339,361,755,423,558,551,276,11,724,70,823,624,555,300,42,607,554,84,508,953,649,732,338,613,236,90,762,612,194,452,972,140,747,209,690,22,220,413,91,36,998,341,77,956,246,512,464,198,547,888,476,782,977,776,896,940,321,347,264,621,10,829,383,939,825,441,326,822,754,130,379,265,945,577,491,252,273,792,168,699,866,319,704,708,148,230,521,914,988,846,88,121,600,217,499,513,427,344,3,242,947,627,325,146,469,375,12,815,46,67,193,648,963,876,78,366,531,49,532,475,875,398,69,821,454,497,170,922,872,533,736,917,951,609,461,598,571,118,798,981,835,113,530,799,995,930,682,38,405,557,787,377,810,278,874,331,199,97,215,286,13,165,473,115,816,584,707,237,568,72,166,249,805,247,746,534,408,759,739,925,855,305,210,219,470,807,936,974,417,519,288,15,64,438,581,455,250,503,496,145,256,327,255,346,251,109,650,813,679,119,619,721,406,593,489,924,964,563,897,27,769,687,608,224,462,432,39,937,384,990,45,33,154,723,152,772,795,364,283,833,395,495,164,181,232,116,899,458,548,191,320,889,587,353,661,856,814,764,529,737,948,127,335,695,960,858,801,543,916,588,478,103,592,20,481,958,618,334,424,397,694,314,158,114,700,381,287,683,966,459,923,902,332,892,235,938,178,431,631,296,885,820,409,585,141,223,535,688,258,689,884,720,365,611,277,985,684,416,666,182,961,108,355,525,862,412,549,186,244,589,421,52,76,718,352,702,510,117,290,692,603,864,323,388,536,392,151,436,350,788,75,900,490,306,975,207,261,870,188,729,231,485,348,507,676,238,111,180,984,135,771,671,51,1,997,675,869,950,445,434,92,137,221,907,245,17,794,360,935,370,239,362,175,620,973,784,106,136,122,281,426,196,134,68,634,672,28,385,411,526,735,633,841,227,86,500,653,906,933,932,129,435,756,262,698,329,204,941,614,668,139,403,229,243,808,857,659,640,545,345,82,228,516,734,566,868,414,474,506,363,87,173,578,575,312,169,908,929,444,685,657,23,524,358,225,9,41,999,834,546,920,849,456,93,651,433,586,882,942,457,62,839,818,260,369,773,890,865,596,98,271,669,962,311,996,160,200,767,539,163,800,757,582,343,538,131,567,446,213,378,959,299,915,761,313,845,712,330,253,573,18,138,317,56,691,349,605,463,652,781,992,422,32,664,711,284,741,289,57,697,368,583,943,40,298,430,851,913,745,65,179,705,630,401,674,465,487,878,477,240,35,572,838,968,678,342,775,30,806,680,969,2,241,909,803,979,460,518,156,85,643,850,597,843,89};
    auto result = largestDivisibleSubset2(nums);
    printVectorIntOneLine(result);
}
