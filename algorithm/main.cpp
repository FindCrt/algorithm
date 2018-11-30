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

#define LRUCache(c) LRUCache cache(c);
int main(int argc, const char * argv[]) {

    string A = "ABCD";
    string B = "CBCE";
    auto result = longestCommonSubstring(A, B);
    printf("%d\n",result);
}
