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

/*
 1. BFS
 2. 暴力解，和1一样，复杂度为O(K*N^2)，K为房子的总数，房子密集时恐怖
 3. 分别求x和y的轴的距离，每个轴的求解，可以建一个一维数组，统计当前行或列上的房子个数，以这个一维数组就可以求解
 4. 同上，但求解单轴时，使用递推方法，上面时间复杂度O(n^2),这里是O(N)
 5. 求中卫数，对单轴而言，最近的距离就是中位数，但中位数位置可能有房子，所以还需要从中心向外扩散，复杂度为O(lgN)*k,k看房子是否集中。
 */
int shortestDistance(vector<vector<int>> &grid) {
    int height = (int)grid.size(), width = (int)grid.front().size();
    vector<Point> houses;
    
    for (int i = 0; i<height; i++) {
        for (int j = 0; j<width; j++) {
            //每个房子为起点进行广度搜索，把它到每个空地的距离求和
            if (grid[i][j]==1) {
                houses.push_back({i,j});
            }
        }
    }
    
    int size = (int)houses.size();
    sort(houses.begin(), houses.end(), Point::compareX);
    int destX;
    if (!(size&1)) { //偶数
        destX = (houses[size/2-1].x+houses[size/2].x)/2;
    }else{
        destX = houses[size/2].x;
    }
    
    sort(houses.begin(), houses.end(), Point::compareY);
    int destY;
    if (!(size&1)) { //偶数
        destY = (houses[size/2-1].y+houses[size/2].y)/2;
    }else{
        destY = houses[size/2].y;
    }
    
    int offset = 0, offsetX;
    
    
    return minDis;
}

int main(int argc, const char * argv[]) {
    uint64_t start = mach_absolute_time();

    vector<vector<int>> grid;
    string path = "/Users/apple/Desktop/short_data";
    read2DVectorInt(path, grid);
    auto result = shortestDistance(grid);
    printf("%d\n",result);
    
    uint64_t duration = mach_absolute_time() - start;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
    printf("exe time: %.1f ms\n",time);
}
