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

#define LRUCache(c) LRUCache cache(c);
int main(int argc, const char * argv[]) {

    vector<string> expression = {"2","*","6","-","(","23","+","7",")","/","(","1","+","2",")"};
//    vector<string> expression = {"(","10","-","7",")","/","3"};
    auto result = build(expression);
    
}
