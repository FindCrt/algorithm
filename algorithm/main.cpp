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

bool intGridVisit(vector<vector<int>> &grid, Point curP, int depth){
    printf("%d\n",grid[curP.x][curP.y]);
    return true;
}

template <class T>
class GridIter{
    
public:
    typedef bool (*VisitPointFunc)(vector<vector<T>> &grid, Point curP, int depth);
    VisitPointFunc visitFunc;
    GridIter(VisitPointFunc visitFunc):visitFunc(visitFunc){};
};

//这样F第三个参数可以接受“任意可以转换了GridIter<T>”
template<class T, class F>
static void BFSGrid(vector<vector<T>> &grid, Point start, F visitFunc){
    BFSGrid(grid, start, (GridIter<T>)visitFunc);
}

template<class T>
static void BFSGrid(vector<vector<T>> &grid, Point &start, GridIter<T> iter){
    queue<Point> pointsQ;
    pointsQ.push(start);
    
    
}

int main(int argc, const char * argv[]) {
    uint64_t start = mach_absolute_time();
    

    vector<vector<int>> grid = {{1,2},{3,4}};
    
//    GridIter<int>::VisitPointFunc func = intGridVisit;
    GridIter<int> iter = intGridVisit;
    
    BFSGrid(grid, {0,0}, intGridVisit);
    
    
    
    
    uint64_t duration = mach_absolute_time() - start;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
    printf("exe time: %.1f ms\n",time);
}
