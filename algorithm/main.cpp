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

#include "TFSort.h"
#include "MinStack.hpp"

#include "page1.hpp"
#include "page2.hpp"
#include "TopK.hpp"
#include "Graph.hpp"
#include "MultiwayTree.hpp"
#include "BinaryTree.hpp"
#include "LFUCache.hpp"

using namespace std;

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

#define LFUCache(n) auto cache = LFUCache(n);
#define set(a,b) cache.set(a,b);
#define get(a) printf("get(%d) %d\n",a,cache.get(a));

int main(int argc, const char * argv[]) {
    
    LFUCache(3)
    set(1, 10)
    set(2, 20)
    set(3, 30)
    get(1)
    set(4, 40)
    get(4)
    get(3)
    get(2)
    get(1)
    set(5, 50)
    get(1)
    get(2)
    get(3)
    get(4)
    get(5)
}
