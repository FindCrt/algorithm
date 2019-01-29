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
#include <queue>
#include <list>
#include <set>
#include <unordered_map>
#include <iostream>
#include <mach/mach_time.h>
#include <unordered_set>
#include <fstream>
#include <stdlib.h>
#include "Trie.hpp"

#include "CommonStructs.hpp"
#include "TypicalProblems.hpp"
#include "TestFuncs.hpp"



int maxSubArrayLen(vector<int> &nums, int k) {
    int sum = 0;
    int idx = 0;
    unordered_map<int, pair<int, int>> sumIdx;
    sumIdx[0] = {-1,-1};
    for (auto &n : nums) {
        sum += n;
        auto i = sumIdx.find(sum);
        if (i == sumIdx.end()) {
            sumIdx[sum] = {idx,idx};
        }else{
            i->second.first = min(i->second.first, idx);
            i->second.second = max(i->second.second, idx);
        }
        idx++;
    }
    
    int maxRange = 0;
    for (auto &p:sumIdx){
        auto i = sumIdx.find(p.first+k);
        if (i != sumIdx.end()) {
            maxRange = max(maxRange, i->second.second-p.second.first);
        }
    }
    
    return maxRange;
}

bool canPermutePalindrome(string &s) {
    bool pair[256];
    memset(pair, 0, sizeof(pair));
    
    for (auto &c : s){
        pair[c] = !pair[c];
    }
    
    int singleCount = 0;
    for (int i = 0; i<256;i++){
        singleCount += pair[i];
    }
    return singleCount<2;
}

class DataStream {
    struct ListNode{
        int val;
        ListNode *next = nullptr;
    };
    ListNode head;
    ListNode *tail = new ListNode();
    unordered_map<int, ListNode*> uniqueNodes;
public:
    
    DataStream(){
        head.next = tail;
    }
    
    void add(int num) {
        auto i = uniqueNodes.find(num);
        if (i == uniqueNodes.end()) {
            auto newNode = new ListNode();
            tail->val = num;
            tail->next = newNode;
            uniqueNodes[num] = tail;
            tail=newNode;
        }else{
            if (i->second->next == tail) {
                tail = i->second;
            }else{
                auto next = i->second->next;
                i->second->val = next->val;
                i->second->next = next->next;
                
                delete next;
                uniqueNodes[i->second->val] = i->second;
            }
            
            uniqueNodes.erase(i);
        }
    }

    int firstUnique() {
//        cout<<head.next<<" "<<head.next->val<<endl;
        return head.next->val;
    }
};

int main(int argc, const char * argv[]) {
    uint64_t start = mach_absolute_time();
    
    
    DataStream ds;
    ds.add(1);
    ds.add(2);
    ds.add(3);
    ds.add(4);
    ds.add(5);
    ds.firstUnique();
    ds.add(1);
    ds.firstUnique();
    ds.add(2);
    ds.firstUnique();
    ds.add(3);
    ds.firstUnique();
    ds.add(4);
    ds.firstUnique();
    ds.add(5);
    
    
    
    uint64_t duration = mach_absolute_time() - start;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
    printf("exe time: %.1f ms\n",time);
}
