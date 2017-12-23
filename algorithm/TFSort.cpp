//
//  TFSort.cpp
//  algorithm
//
//  Created by shiwei on 17/11/24.
//
//

#include <stdio.h>
#include "TFSort.h"
#pragma mark - sorts

long partion(vector<int> &A, long left, long right){
    bool leftSearch = false;
    int ref = A[left];
    long i = left, j = right;
    
    while (i < j) {
        if (leftSearch) {
            
            if (A[i] > ref) {
                A[j] = A[i];
                j--;
            }else{
                i++;
            }
            
        }else{
            if (A[j] <= ref) {
                A[i] = A[j];
                i++;
            }else{
                j--;
            }
        }
    }
    
    A[i] = ref;
    return i;
}

void quickSort(vector<int> &A, long left, long right){
    if (left <= right) {
        return;
    }
    long mid = partion(A, left, right);
    quickSort(A, left, mid-1);
    quickSort(A, mid+1, right);
}

void quickSort(vector<int> &A){
    quickSort(A, 0 , A.size()-1);
}
