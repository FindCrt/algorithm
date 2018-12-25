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

int singleNumber(vector<int> &A) {
    int num = 0;
    for (auto &n : A){
        num ^= n;
    }
    
    return num;
}

vector<int> getNarcissisticNumbers(int n) {
    if (n == 1) {
        return {1,2,3,4,5,6,7,8,9};
    }
    
    vector<int> result;
    int pows[10];
    for (int i = 0; i<10; i++) {
        pows[i] = pow(i, n);
    }
    
    int min = pow(10, n-1);
    int max = pow(10, n);
    int digits[n];
    memset(digits, 0, sizeof(digits));
    digits[0] = 1;
    
    //除法特别慢，计算机没法优化，用加法
    for (int k = min+1; k<max; k++) {
        bool carry = 1;
        for (int i = n-1; i>=0 && carry; i--) {
            if (digits[i] == 9) {
                digits[i] = 0;
            }else{
                digits[i]++;
                carry = 0;
            }
        }
        
        int powSum = 0;
        for (int i = 0; i<n; i++) {
            powSum += pows[digits[i]];
        }
        
        if (powSum == k) {
            result.push_back(powSum);
        }
    }
    
    return result;
}

///1与，对方；1或，都是1；0与，都是0；0或，对方
int aplusb(int a, int b) {
    int result = 0;
    bool carry = 0;
    for (int i = 0; i<sizeof(int)*8; i++) {
        int digitA = a &1;
        int digitB = b &1;
        
        bool one = false;
        if (digitA ^ digitB) {
            if (!carry) {
                one = true;
            }
        }else{
            if (digitA) {
                if (carry) {
                    one = true;
                }
                carry = true;
            }else{
                if (carry) {
                    one = true;
                    carry = false;
                }
            }
        }
        
        if (one) {
            result |= (1<<i);
        }
        
        a >>= 1;
        b >>= 1;
    }
    
    return result;
}

void rotateString(string &str, int start, int end){
    int i = start, j = end;
    while (i<j) {
        char temp = str[i];
        str[i] = str[j];
        str[j] = temp;
        
        i++;
        j--;
    }
}

void rotateString(string &str, int offset) {
    int len = (int)str.length();
    if (len == 0) {
        return;
    }
    offset %= len;
    if (offset == 0) {
        return;
    }
    
    rotateString(str, 0, len-offset-1);
    rotateString(str, len-offset, len-1);
    rotateString(str, 0, len-1);
}


vector<int> twoSum(vector<int> &numbers, int target) {
    
    vector<int> result;
    unordered_map<int, int> hash;
    for (int i = 0; i < numbers.size(); ++i) {
        if (hash.find(target - numbers[i]) != hash.end()) {
            result.push_back(hash[target - numbers[i]]);
            result.push_back(i);
            break;
        }
        hash[numbers[i]] = i;
    }
    
    return result;
}

static int XORMerge(int &a, int &b){
    return a^b;
}

vector<int> intervalXOR(vector<int> &A, vector<Interval> &query) {
    typedef TFDataStruct::SegmentTree<int, XORMerge> SegmentTree;
    auto root = SegmentTree::build(A);
    vector<int> result;
    for (auto &qu : query){
        result.push_back(SegmentTree::query(root, qu.start, qu.end+qu.start-1));
    }
    
    return result;
}

//q(n*a1-n+n*(n-1)/2)
int getSum(int A, int B) {
    int start = (A+2)/3;
    int end = B/3;
    
    int count = end-start+1;
    return 3*(count*start+count*(count-1)/2);
}

char firstUniqChar(string &str) {
    unordered_map<char, int> mark;
    for (auto &c : str){
        mark[c]++;
    }
    
    for (auto &c : str){
        if (mark[c] == 1) {
            return c;
        }
    }
    
    return '\0';
}

bool isUnique(string &str) {
    bool mark[256];
    memset(mark, 0, sizeof(mark));
    
    for (auto &c : str){
        if (mark[c]) {
            return false;
        }else{
            mark[c] = true;
        }
    }
    
    return true;
}

int strStr(string &source, string &target) {
    int len1 = (int)source.length(), len2 = (int)target.length();
    for (int x = 0; x<=len1-len2; x++) {
        int i = x, j = 0;
        while (i< len1 && j<len2 && source[i] == target[j]) {
            i++;
            j++;
        }
        if (j == len2) {
            return x;
        }
    }
    return -1;
}

bool isLegalIdentifier(string &str) {
    if (str.length() == 0) {
        return true;
    }
    if (str.front()>='0' && str.front() <= '9') {
        return false;
    }
    
    for (auto &c : str){
        if (c =='_' ||
            (c<='9' && c >= '0') ||
            (c<='z' && c >= 'a') ||
            (c<='Z' && c >= 'A')) {
            continue;
        }else{
            return false;
        }
    }
    
    return true;
}

void sortIntegers(vector<int> &A) {
    for (int i = 1; i<A.size(); i++) {
        int cur = A[i];
        int j = i-1;
        while (j>=0 && A[j]>cur) {
            A[j+1] = A[j];
            j--;
        }
        A[j+1] = cur;
    }
}

static bool intervalComp(Interval &a, Interval &b){
    return a.start < b.start;
}
bool canAttendMeetings(vector<Interval> &intervals) {
    sort(intervals.begin(), intervals.end(), intervalComp);
    
    auto &lastInterval = intervals.front();
    for (int i = 1; i< intervals.size(); i++){
        if (intervals[i].start<=lastInterval.end) {
            return false;
        }
        lastInterval = intervals[i];
    }
    
    return true;
}

int deduplication(vector<int> &nums) {
    if (nums.size() < 2) {
        return (int)nums.size();
    }
    
    unordered_map<int, bool> exist;
    exist[nums.front()] = true;
    
    int i = 1, j = 1;
    while (j < nums.size()) {
        if (exist.find(nums[j]) == exist.end()) {
            nums[i]=nums[j];
            exist[nums[j]] = true;
            i++;
        }
        j++;
    }
    
    return i;
}

///最小堆，然后求和
string depress(int m, int k, vector<int> &arr) {
    //维持k个最小值s，使用最大堆
    TFDataStruct::heap<int> minHeap(false, k);
    
    for (auto &num : arr){
        if (!minHeap.isFull()) {
            minHeap.append(num);
        }else{
            cout<<minHeap<<endl;
            if (num < minHeap.getTop()) {
                minHeap.replaceTop(num); //跟新顶部
            }
        }
    }
    
    cout<<minHeap<<endl;
    
    int sum = 0;
    while (!minHeap.isEmpty()) {
        sum += minHeap.popTop();
    }
    
    return sum<m?"yes":"no";
}

int partion(vector<int> &a, int start, int end){
    int i = start, j = end;
    
    int st = a[start];
    bool left = false;
    while (i < j) {
        if (left) {
            if (a[i] > st) {
                a[j] = a[i];
                j--;
                left = !left;
            }else{
                i++;
            }
        }else{
            if (a[j] <= st) {
                a[i] = a[j];
                i++;
                left = !left;
            }else{
                j--;
            }
        }
    }
    
    a[i] = st;
    return i;
}

//思路归纳： 一个查找区间，目标确定在这个区间内，将区间划分成两部分，判定区间在左边还是在右边，然后就可以缩小查找区间。循环不变体是：目标在查找区间内；变化过程是：区间的二分切割；退出条件时：区间缩小到长度为1.
//再高一层的抽象为：大问题化小，小到一定程度，问题便非常简单可解了。
int getAns(vector<int> &a) {
    if (a.empty()) {
        return -1;
    }
    int mid = (a.size()-1)/2;
    vector<int> copy = a;
    int left = 0, right = a.size()-1;
    while (left < right) {
        int k = partion(copy, left, right);
        printVectorOneLine(copy);
        printf("k=%d\n",k);
        if (k==mid) {
            left = right = k;
            break;
        }else if (k < mid) {
            left = k+1;
        }else{
            right = k-1;
        }
    }
    
    int find = copy[left];
    for (int i = 0; i<a.size(); i++) {
        if (a[i] == find) {
            return i;
        }
    }
    
    return -1;
}

int maximumSwap(int num) {
    if (num < 10) {
        return num;
    }
    vector<int> digits;
    int num2 = num;
    while (num2 > 0) {
        digits.push_back(num2%10);
        num2 /= 10;
    }
    
    printVectorOneLine(digits);
    
    int maxIdx = 0, maxNum = digits.front();
    int change1 = -1, change2 = -1;
    for (int i = 1; i<digits.size(); i++) {
        if (digits[i]>maxNum) {
            maxNum = digits[i];
            maxIdx = i;
        }else if (digits[i] < maxNum){
            change1 = maxIdx;
            change2 = i;
        }
    }
    
    if (change1 >= 0) {
        int temp = digits[change1];
        digits[change1] = digits[change2];
        digits[change2] = temp;
    }
    
    int result = 0, weight = 1;
    for (auto &d : digits){
        result += d*weight;
        weight *= 10;
    }
    
    return result;
}

#define LRUCache(c) LRUCache cache(c);
int main(int argc, const char * argv[]) {

//    vector<int> nums = {101,527,373,526,199,938,915,766,429,951};
    
    auto result = maximumSwap(0);
//    printVectorOneLine(nums);
//    printf("%d \n",result);
    cout<<result<<endl;
    
//    sort(nums.begin(), nums.end());
//    printVectorOneLine(nums);
}
