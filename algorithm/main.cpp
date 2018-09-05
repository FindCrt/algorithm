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

#include "TopK.hpp"
#include "Graph.hpp"
#include "MultiwayTree.hpp"
#include "BinaryTree.hpp"
#include "LFUCache.hpp"
#include "SegmentTree.hpp"
#include "CommonStructs.hpp"

#define LFUCache(n) auto cache = LFUCache(n);
#define set(a,b) cache.set(a,b);
#define get(a) printf("get(%d) %d\n",a,cache.get(a));

vector<int> resolveSegmentInput(string source, int size){
    vector<int> counts(size, 0);
    int mark = 0, firstNum = -1;
    bool isLeave = false;
    for (int i = 0; i<source.size(); i++) {
        char c = source[i];
        if (c == '[') {
            mark = i+1;
        }else if (c == ','){
            int num = atoi(source.substr(mark,i-mark).c_str());
            if (firstNum < 0) {
                firstNum = num;
                mark = i+1;
            }else if (firstNum == num){
                isLeave = true;
            }
        }else if (c == '='){
            mark = i+1;
        }else if (c == ']'){
            int num = atoi(source.substr(mark,i-mark).c_str());
            if (isLeave) {
                counts[firstNum] = num;
            }
            firstNum = -1;
        }
    }
    
    return counts;
}

//833. 进程序列
vector<int> numberOfProcesses(vector<Interval> &logs, vector<int> &queries) {
    // Write your code here
    vector<pair<long, long>> vec;
    for(auto e: logs)
    {
        vec.push_back(make_pair(e.start, 0));
        vec.push_back(make_pair(e.end+1, 1));
    }
    
    for(int i=0; i<queries.size(); i++)
    {
        vec.push_back(make_pair(queries[i], 2));
    }
    
    int count=0;
    vector<int> res(queries.size());
    sort(vec.begin(), vec.end());
    map<int, int> mp;
    
    
    for(int i=0; i<vec.size(); i++)
    {
        if(vec[i].second==0)
        {
            count++;
        }
        else if (vec[i].second==1)
        {
            count--;
        }
        else
        {
            mp[vec[i].first] = count;
        }
    }
    int i=0;
    for(auto e: queries)
    {
        res[i++] = mp[e];
    }
    return res;
}

//751. 约翰的生意 这题符合线段树的应用环境
//借一位刷题朋友的话：线段树的应用环境是构建时使用点粒度，查询时使用区间粒度。
//这题不仅应用环境很匹配，而且数据量也很大，别说暴力解不行，就是递归实现的线段树都不行
vector<int> business(vector<int> &A, int k) {
    typedef SegmentTree<int, minMergeFunc> MyTree;
    
    auto root = MyTree::build(A);
    vector<int> result;
    
    int size = (int)A.size();
    for (int i = 0; i<size; i++) {
        int minPrice = MyTree::query(root, max(i-k, 0), min(size, i+k), INT_MIN);
        result.push_back(max(0, A[i]-minPrice));
    }
    
    return result;
}

//205. 区间最小数
vector<int> intervalMinNumber(vector<int> &A, vector<Interval> &queries) {
    typedef SegmentTree<int, minMergeFunc> MyTree;
    
    auto root = MyTree::build(A);
    vector<int> result;
    
    for (auto &inter : queries) {
        int minNum = MyTree::query(root, inter.start, inter.end, INT_MIN);
        result.push_back(minNum);
    }
    
    return result;
}

//206. 区间求和 I
vector<long long> intervalSum(vector<int> &A, vector<Interval> &queries) {
    typedef SegmentTree<long long, sumMergeFunc, int> MyTree;
    
    MyTree::NodeType *root = MyTree::build(A);
    vector<long long> result;
    
    for (auto &inter : queries) {
        long long sum = MyTree::query(root, inter.start, inter.end, 0);
        result.push_back(sum);
    }
    
    return result;
}

class AnimalShelter {
    const int type_dog = 1;
    const int type_cat = 0;
    struct Animal{
        string name;
        int type;
        Animal(){
            name = "";
            type = -1;
        };
        Animal(string &name, int type):name(name),type(type){};
        
        Animal *pre = nullptr;
        Animal *next = nullptr;
        Animal *pre_sameType = nullptr;
        Animal *next_sameType = nullptr;
    };
    
    Animal *anyHead = new Animal();
    Animal *anyTail = new Animal();
    Animal *catHead = new Animal();
    Animal *catTail = new Animal();
    Animal *dogHead = new Animal();
    Animal *dogTail = new Animal();
    
    //node1->nodex ==> node1->node2->nodex
    inline void insert(Animal *node1, Animal *node2){
        node1->next->pre = node2;
        node2->next = node1->next;
        
        node1->next = node2;
        node2->pre = node1;
    }
    inline void insert_sameType(Animal *node1, Animal *node2){
        node1->next_sameType->pre_sameType = node2;
        node2->next_sameType = node1->next_sameType;
        
        node1->next_sameType = node2;
        node2->pre_sameType = node1;
    }
    
    //nodex->node1->node2 ==> nodex->node2
    inline void remove(Animal *node1, Animal *node2){
        node1->pre->next = node2;
        node2->pre = node1->pre;
        
        node1->pre = nullptr;
        node1->next = nullptr;
    }
    inline void remove_sameType(Animal *node1, Animal *node2){
        node1->pre_sameType->next_sameType = node2;
        node2->pre_sameType = node1->pre_sameType;
        
        node1->pre_sameType = nullptr;
        node1->next_sameType = nullptr;
    }
 
public:
    
    AnimalShelter(){
        anyTail->next = anyHead;
        anyHead->pre = anyTail;
        
        catTail->next_sameType = catHead;
        catHead->pre_sameType = catTail;
        
        dogTail->next_sameType = dogHead;
        dogHead->pre_sameType = dogTail;
    }

    void enqueue(string &name, int type) {
        auto newAni = new Animal(name, type);
        
        insert(anyTail, newAni);
        
        if (type == type_cat) {
            insert_sameType(catTail, newAni);
        }else{
            insert_sameType(dogTail, newAni);
        }
    }
    
    string dequeueAny() {
        if (anyHead==anyTail->next) {
            return "";
        }
        
        auto node = anyHead->pre;
        remove(anyHead->pre, anyHead);
        
        if (node->type == type_cat) {
            remove_sameType(catHead->pre_sameType, catHead);
        }else{
            remove_sameType(dogHead->pre_sameType, dogHead);
        }
        
        auto result = node->name;
        delete node;
        return result;
    }

    string dequeueDog() {
        if (dogHead==dogTail->next) {
            return "";
        }
        
        auto node = dogHead->pre_sameType;
        remove_sameType(dogHead->pre_sameType, dogHead);
        
        remove(node, node->next);
        
        auto result = node->name;
        delete node;
        return result;
    }

    string dequeueCat() {
        if (catHead==catTail->next) {
            return "";
        }
        
        auto node = catHead->pre_sameType;
        remove_sameType(catHead->pre_sameType, catHead);
        
        remove(node, node->next);
        
        auto result = node->name;
        delete node;
        return result;
    }
    
    void showAnimals(){
        printf("\n*******************\nany: ");
        auto cur = anyTail->next;
        while (cur != anyHead) {
            cout<<cur->name<<" ";
            cur = cur->next;
        }
        printf("\n");
        
        printf("cat: ");
        cur = catTail->next_sameType;
        while (cur != catHead) {
            cout<<cur->name<<" ";
            cur = cur->next_sameType;
        }
        printf("\n");
        
        printf("dog: ");
        cur = dogTail->next_sameType;
        while (cur != dogHead) {
            cout<<cur->name<<" ";
            cur = cur->next_sameType;
        }
        printf("\n");
    }
};

#define enqueue(a,b) {string name = a;shelter.enqueue(name, b);}shelter.showAnimals();
#define dequeueAny() cout<<shelter.dequeueAny()<<endl;shelter.showAnimals();
#define dequeueDog() cout<<shelter.dequeueDog()<<endl;shelter.showAnimals();
#define dequeueCat() cout<<shelter.dequeueCat()<<endl;shelter.showAnimals();

int main(int argc, const char * argv[]) {
    AnimalShelter shelter;
    enqueue("ajpy", 1)
    enqueue("wajb", 0)
    dequeueAny()
    enqueue("hjyw", 1)
    dequeueAny()
    enqueue("wtyw", 1)
    enqueue("jght", 1)
    enqueue("apwy", 0)
    dequeueCat()
    dequeueDog()
    enqueue("ybwg", 0)
    enqueue("jpwa", 1)
    dequeueCat()
    dequeueDog()
    enqueue("jayh", 1)
    enqueue("atww", 0)
    dequeueDog()
    enqueue("wjpt", 0)
    dequeueCat()
    dequeueDog()
    enqueue("yhwp", 0)
    enqueue("gwya", 1)
    dequeueCat()
    dequeueCat()
    enqueue("jgwb", 0)
    enqueue("agyp", 1)
    dequeueDog()
    dequeueCat()
    enqueue("pbtw", 1)
    dequeueDog()
    enqueue("wgjy", 0)
    enqueue("gbat", 0)
    dequeueAny()
    enqueue("ahbw", 0)
    dequeueDog()
    dequeueCat()
    dequeueCat()
    enqueue("wbya", 0)
    dequeueCat()
    enqueue("pgty", 0)
}
