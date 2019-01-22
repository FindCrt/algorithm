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
#include <set>
#include <unordered_map>
#include <iostream>
#include <mach/mach_time.h>
#include <unordered_set>
#include <fstream>
#include <stdlib.h>

#include "CommonStructs.hpp"
#include "TypicalProblems.hpp"
#include "TestFuncs.hpp"

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

#define add(i) ts.add(i);
#define find(i) {printf("%s\n",ts.find(i)?"true":"false");}
int main(int argc, const char * argv[]) {
    uint64_t start = mach_absolute_time();

    vector<string> names = {"James", "james", "Bill Gates", "bill Gates", "Hello World", "HELLO WORLD", "Helloworld"};
    auto result = nameDeduplication(names);
    printVectorOneLine(result);
    
    
    uint64_t duration = mach_absolute_time() - start;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
    printf("exe time: %.1f ms\n",time);
}
