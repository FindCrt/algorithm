//
//  CommonStructs.cpp
//  algorithm
//
//  Created by shiwei on 2018/9/1.
//

#include "CommonStructs.hpp"
#include <fstream>
#include <assert.h>
#include <math.h>

void readFile(string &path, const function<void(string &)>& handleLine){
    assert(handleLine);
    
    ifstream infile(path.c_str());
    if (!infile.is_open()){
        return;
    }
    
    string s;
    while (getline(infile, s)) {
        handleLine(s);
    }
    infile.close();
}


#pragma mark - 经典常用函数

/** 是否素数 */
bool isPrimeNum(int num){
    if (num == 2 || num == 3) {
        return true;
    }
    if (num%6 != 1 && num%6 != 5) {
        return false;
    }
    
    int temp = sqrt(num);
    for (int i = 5; i<temp+1; i++) {
        if (num%i == 0) {
            return false;
        }
    }
    
    return true;
}

int partion(vector<int> &nums, int start, int end){
    /*
     //使用随机参照
     int refIdx = random()%(end-start+1)+start;
     int ref = nums[refIdx]; //参照物
     nums[refIdx]=nums[start];
     */
    
    int ref = nums[start+(end-start)/2]; //参照物
    int i = start, j = end;
    bool left = false;
    
    while (i < j) {
        if (left) {
            if (nums[i]>ref) {
                nums[j] = nums[i];
                j--;
                left = !left;
            }else{
                i++;
            }
        }else{
            if (nums[j]<=ref) {
                nums[i] = nums[j];
                i++;
                left = !left;
            }else{
                j--;
            }
        }
    }
    
    nums[i] = ref;
    return i;
}

int kthSmallest(int k, vector<int> &nums) {
    int start = 0, end = (int)nums.size()-1;
    while (start<end) {
        int x = partion(nums, start, end);
        printf("x: %d\n",x);
        if (x == k-1) {
            return nums[x];
        }else if (x < k-1){
            start = x+1;
        }else{
            end = x-1;
        }
    }
    
    return nums[start];
}

int kthLargestElement(int n, vector<int> &nums){
    return kthSmallest((int)nums.size()-n+1, nums);
}

vector<int> searchRange(vector<int> &A, int target){
    
    int left = -1, right = (int)A.size();
    int find = -1;
    while (left < right-1) {
        int mid = left+(right-left)/2;
        if (A[mid] == target) {
            find = mid;
            break;
        }else if (A[mid] < target){
            left = mid;
        }else{
            right = mid;
        }
    }
    
    if (find < 0) {
        return {-1, -1};
    }
    
    int midLeft = find;
    while (left < midLeft-1) {
        int mid = left+(midLeft-left)/2;
        if (A[mid] == target) {
            midLeft = mid;
        }else{
            left = mid;
        }
    }
    
    int midRight = find;
    while (midRight < right-1) {
        int mid = midRight+(right-midRight)/2;
        if (A[mid] == target) {
            midRight = mid;
        }else{
            right = mid;
        }
    }
    
    return {midLeft, midRight};
}
