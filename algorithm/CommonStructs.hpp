//
//  CommonStructs.hpp
//  algorithm
//
//  Created by shiwei on 2018/9/1.
//

#ifndef CommonStructs_hpp
#define CommonStructs_hpp

#include <stdio.h>
#include <vector>
#include <iostream>
#include <functional>
#include "BinaryTree.hpp"


using namespace std;

class Interval {
public:
    int start, end;
    Interval(int start, int end) {
        this->start = start;
        this->end = end;
    }
};


#pragma mark - 调试函数

template<class T>
static void printVector(vector<T> &vector){
    for (auto &val : vector){
        cout<<val<<endl;
    }
}

template<class T>
static void printVectorOneLine(vector<T> &vector){
    for (auto &val : vector){
        cout<<val<<" ";
    }
    cout<<endl;
}

template<class T>
static void printTwoDVector(vector<vector<T>> & twoDVector){
    for (auto iter = twoDVector.begin(); iter != twoDVector.end(); iter++) {
        printVectorOneLine(*iter);
    }
}

#pragma mark - 辅助工具函数

extern void readFile(string &path, const function<void(string &)>& handleLine);

#endif /* CommonStructs_hpp */
