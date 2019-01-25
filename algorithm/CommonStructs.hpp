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
#include <queue>


using namespace std;

#define FirstSmall(first,second)\
if (first>second){auto temp=first;first=second;second=temp;}

class Interval {
public:
    int start, end;
    Interval(int start, int end) {
        this->start = start;
        this->end = end;
    }
};

struct Point{
    int x;
    int y;
    
    static bool compareX(Point &p1, Point &p2){
        return p1.x<p2.x;
    }
    
    static bool compareY(Point &p1, Point &p2){
        return p1.y<p2.y;
    }
};

class ListNode {
public:
    int val;
    ListNode *next;
    ListNode(int val) {
        this->val = val;
        this->next = NULL;
    }
    
    static ListNode *createList(vector<int> vals){
        ListNode *head = nullptr;
        ListNode *last = nullptr;
        for (auto iter = vals.begin(); iter != vals.end(); iter ++) {
            ListNode *node = new ListNode(*iter);
            
            if (head == nullptr) {
                head = node;
            }else{
                last->next = node;
            }
            last = node;
        }
        
        return head;
    }
    
    static void printList(ListNode *list){
        if (list == nullptr) {
            printf("list is empty!\n");
            return;
        }
        
        printf("%d",list->val);
        list = list->next;
        
        while (list) {
            printf("->%d",list->val);
            list = list->next;
        }
        
        printf("\n");
    }
};


#pragma mark - 调试函数

#define printBool(x) printf("%s\n",x?"True":"False");

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
static void printArrayOneLine(T *array, int size){
    for (int i =0; i<size; i++){
        cout<<array[i]<<" ";
    }
    cout<<endl;
}


template<class T>
static void printTwoDVector(vector<vector<T>> & twoDVector){
    for (auto iter = twoDVector.begin(); iter != twoDVector.end(); iter++) {
        printVectorOneLine(*iter);
    }
}

template<class T1, class T2>
static void printVectorPair(vector<pair<T1, T2>> &vector){
    for (auto &p : vector){
        cout<<"("<<p.first<<","<<p.second<<") ";
    }
    cout<<endl;
}

#pragma mark - 辅助工具函数

extern void readFile(string &path, const function<void(string &)> handleLine);


#pragma mark - 经典常用函数

/** 是否素数 */
extern bool isPrimeNum(int num);

/**
 * 将数组切割为两部分，使得左边<=中间数<右边，返回这个中间数的索引;
 * 这个函数是快排、求中位数、第k小的数等的辅助工具
 */
extern int partion(vector<int> &nums, int start, int end);

/** 第k小的数 */
extern int kthSmallest(int k, vector<int> &nums);

/** 第k大的数 */
extern int kthLargestElement(int n, vector<int> &nums);

//TODO: topK问题

template<class T>
bool canReachPoint(vector<vector<T>> &map, Point start, T wall, T road, T target, vector<Point> *path = nullptr){
    if (
        start.x<0||
        start.x >= map.size() ||
        start.y <0||
        start.y >= map.front().size() ||
        map[start.x][start.y] == wall) {
        return false;
    }
    
    if (map[start.x][start.y] == target) {
        return true;
    }
    
    map[start.x][start.y] = wall;
    if (path) {
        path->push_back(start);
    }
    if (canReachPoint(map, {start.x+1, start.y}, wall, road, target, path)) {
        return true;
    }
    if (canReachPoint(map, {start.x-1, start.y}, wall, road, target, path)) {
        return true;
    }
    if (canReachPoint(map, {start.x, start.y+1}, wall, road, target, path)) {
        return true;
    }
    if (canReachPoint(map, {start.x, start.y-1}, wall, road, target, path)) {
        return true;
    }
    
    if (path) {
        path->pop_back();
    }
    
    return false;
}

template<class T>
bool canReachPoint(vector<vector<T>> &map, Point start, T wall, T road, Point target){
    if (
        start.x<0||
        start.x >= map.size() ||
        start.y <0||
        start.y >= map.front().size() ||
        map[start.x][start.y] == wall) {
        return false;
    }
    
    if (start.x == target.x && start.y == target.y) {
        return true;
    }
    
    map[start.x][start.y] = wall;
    if (canReachPoint(map, {start.x+1, start.y}, wall, road, target)) {
        return true;
    }
    if (canReachPoint(map, {start.x-1, start.y}, wall, road, target)) {
        return true;
    }
    if (canReachPoint(map, {start.x, start.y+1}, wall, road, target)) {
        return true;
    }
    if (canReachPoint(map, {start.x, start.y-1}, wall, road, target)) {
        return true;
    }
    
    return false;
}



//#define swapArray()

template<class _T1, class _T2>
class PairSort{
public:
    static bool pairFirstComp(pair<_T1, _T2> &pair1, pair<_T1, _T2> &pair2){
        return pair1.first<pair2.first;
    }
    
    static bool pairSecondComp(pair<_T1, _T2> &pair1, pair<_T1, _T2> &pair2){
        return pair1.second<pair2.second;
    }
};

namespace TFDataStruct {
    class StringHash{
        long long p;
        long long mod;
    public:
        StringHash(long long p=1e9+7, long long mod=1e9+9):p(p),mod(mod){}
        long long hash(string &str){
            long long hashVal = 1;
            for (auto &c : str){
                hashVal = (hashVal*p+c)%mod;
            }
            return hashVal;
        }
    };
}

static void splitWords(string &text, vector<string> &words){
#define isabc(x) ((x>='a'&&x<='z')||(x>='A'&&x<='Z'))
    int start=-1, i = 0;
    for (auto &c : text){
        
        if (start<0) {
            if (isabc(c)) {
                start = i;
            }
        }else{
            if (!isabc(c)) {
                words.push_back(text.substr(start, i-start));
                start = -1;
            }
        }
        
        i++;
    }
}


#endif /* CommonStructs_hpp */
