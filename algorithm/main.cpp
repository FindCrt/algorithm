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
#include "page2.hpp"
#include "TopK.hpp"
#include "Graph.hpp"
#include "MultiwayTree.hpp"
#include "BinaryTree.hpp"

using namespace std;

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

int main(int argc, const char * argv[]) {
    
    vector<int> nums = {1,2,2};
    auto result = permuteUnique2(nums);
    printTwoDVector(result);
}
