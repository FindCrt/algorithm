//
//  TypicalProblems.hpp
//  algorithm
//
//  Created by shiwei on 2019/1/2.
//

///具有代表性的一些经典题集合

#ifndef TypicalProblems_hpp
#define TypicalProblems_hpp

#include <stdio.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <stack>
#include <queue>
#include <list>
#include <unordered_map>
#include <iostream>
#include "heap.hpp"
#include <mach/mach_time.h>
#include <unordered_set>
#include <fstream>
#include<stdlib.h>
#include "CommonStructs.hpp"

#pragma mark - 二分查找

/**
 nums需要是递增数组,二分查找目标值，如果没有:
 flag = 0, 返回-1，默认情况
 flag < 0, 返回小于target的第一个数
 flag > 0, 返回大于target的第一个数
 [start, end] 是指定查找的区间，边界都包含
 */
template<class T>
int binaryFind(vector<T> &nums, T target, int flag = 0, int start = -1, int end = -1){
    int i = -1, j = (int)nums.size();  //左 <, 右 >,边界不包含
    if (start>0 && end > 0) {
        i = start-1;
        j = end+1;
    }
    
    while (i<j-1) {
        int mid = i+(j-i)/2; //不使用(i+j)/2是为了防止int越界
        if (nums[mid] == target) {
            return mid;
        }else if (nums[mid] < target){
            i = mid;
        }else{
            j = mid;
        }
    }
    if (flag == 0) {
        return -1;
    }
    return flag<0?i:j;
}

/** 查找一个数在有序数组中的范围，考虑到这个数不存在或重复的情形；这是进化版的二分法 */
vector<int> searchRange(vector<int> &A, int target, int start = -1, int end = -1){
    
    int left = -1, right = (int)A.size();  //左 <, 右 >,边界不包含
    if (start>0 && end > 0) {
        left = start-1;
        right = end+1;
    }
    
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


#endif /* TypicalProblems_hpp */
