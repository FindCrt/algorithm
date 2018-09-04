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

static void printVectorString(vector<string> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]<<" "<<endl;
    }
}

static void printVectorStringOneLine(vector<string> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]<<" ";
    }
    cout<<endl;
}

static void printTwoDVectorString(vector<vector<string>> &vector){
    for (int i = 0; i<vector.size(); i++) {
        printVectorStringOneLine(vector[i]);
    }
}

static void printVectorInt(vector<int> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]<<" "<<endl;
    }
}

template<class T>
static void printVectorIntOneLine(vector<T> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]<<",";
    }
    cout<<endl;
}

static void printVectorNodeOneLine(vector<TreeNode *> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]->val<<" ";
    }
}

static void printTwoDVector(vector<vector<int>> & twoDVector){
    for (auto iter = twoDVector.begin(); iter != twoDVector.end(); iter++) {
        printVectorIntOneLine(*iter);
    }
}

#pragma mark - 辅助工具函数

extern void readFile(string &path, const function<void(string &)>& handleLine);

#endif /* CommonStructs_hpp */
