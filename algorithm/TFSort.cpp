//
//  TFSort.cpp
//  algorithm
//
//  Created by shiwei on 17/11/24.
//
//

#include <stdio.h>
#include "TFSort.h"
#include "CommonStructs.hpp"

void quickSort(vector<int> &A, int left, int right){
    if (left >= right) {
        return;
    }
    int mid = partion(A, left, right);
    quickSort(A, left, mid-1);
    quickSort(A, mid+1, right);
}

void quickSort(vector<int> &A){
    quickSort(A, 0 , (int)A.size()-1);
}
