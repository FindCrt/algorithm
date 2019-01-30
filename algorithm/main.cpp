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

class RandomizedCollection {
    unordered_map<int, unordered_set<int>> idxes;
    vector<int> nums;
    int size = 0;
public:
    
    RandomizedCollection() {
        srandom((int)time(0));
    }
    
    bool insert(int val) {
        idxes[val].insert(size);
        nums.push_back(val);
        
        size++;
        return idxes[val].size()>1;
    }
    
    bool remove(int val) {
        auto &idxSet = idxes[val];
        if (idxSet.empty()) {
            return false;
        }
        
        auto removeIdx = *idxSet.begin();
        idxSet.erase(idxSet.begin());
        
        nums[removeIdx] = nums.back();
        nums.pop_back();
        
        size--;
        if (removeIdx != size) {
            auto last = idxes.find(nums[removeIdx]);
            last->second.erase(size);
            last->second.insert(removeIdx);
        }
        
        return true;
    }
    
    int getRandom() {
        auto idx = random()%nums.size();
        return nums[idx];
    }
};

int findMaxLength(vector<int> &nums) {
    unordered_map<int, int> diffIdx;
    
    int curDiff = 0;
    int idx = 0;
    diffIdx[0] = -1;
    
    int maxLen = 0;
    for (auto &n : nums){
        curDiff += n==0?-1:1;
        auto iter = diffIdx.find(curDiff);
        if (iter == diffIdx.end()) {
            diffIdx[curDiff]=idx;  //值保存第一个，这就是最前面的索引
        }else{
            //后面查找到具有相同差值(1的个数减0的个数)的，它们之间的区间可以构成解，所以更新最大长度
            maxLen = max(maxLen, idx - iter->second);
        }
        idx++;
    }
    return maxLen;
}

vector<string> subdomainVisits(vector<string> &cpdomains) {
    unordered_map<string, int> counts;
    for (auto &dom : cpdomains){
        int ct=0;
        int idx = 0, len = (int)dom.length();
        for (auto &c:dom){
            if (c==' ') {
                ct = rangeStringToInt(dom, 0, idx-1);
                counts[dom.substr(idx+1,len-idx-1)] += ct;
            }else if (c=='.'){
                counts[dom.substr(idx+1,len-idx-1)] += ct;
            }
            
            idx++;
        }
    }
    
    vector<string> result;
    for (auto &p:counts){
        result.push_back(to_string(p.second)+" "+p.first);
    }
    return result;
}

int numRabbits(vector<int> &answers) {
    unordered_map<int, int> groups;
    for (auto &ans:answers){
        groups[ans]++;
    }
    
    int num = (int)answers.size();
    for (auto &p:groups){
        int curNum = p.first+1;
        num += curNum-(p.second-1)%curNum-1;
    }
    return num;
}

int numJewelsInStones(string &J, string &S) {
    bool jewelry[52];
    memset(jewelry, 0, sizeof(jewelry));
    for (auto &c:J){
        int idx = c>'a'?c-'a':c-'A';
        jewelry[idx] = true;
    }
    
    int count = 0;
    for (auto &c:S){
        int idx = c>'a'?c-'a':c-'A';
        count += jewelry[idx];
    }
    
    return count;
}

vector<int> findErrorNums(vector<int> &nums) {
    int idx = 1;
    for (auto &n:nums){
        if (n!=idx) {
            return {n,idx};
        }
        idx++;
    }
    return {};
}

vector<int> smallestRange(vector<vector<int>> &nums) {
    vector<pair<int, int>> allNumbers;
    int idx = 0;
    //TODO: 可以用败者树做归并，少去排序的时间
    for (auto &v:nums){
        for (auto &n:v){
            allNumbers.push_back({n, idx});
        }
        idx++;
    }
    
    sort(allNumbers.begin(), allNumbers.end(), PairSort<int, int>::pairFirstComp);
    
    printVectorPair(allNumbers);
    
    int size = (int)allNumbers.size();
    int maxKind = (int)nums.size();
    int counts[maxKind];
    memset(counts, 0, sizeof(counts));
    int kindCount = 0;
    
    int i = 0, j = 0;
    int minStart=0, minRange = INT_MAX;
    
    while (1) {
        //开头前进，满足条件
        while (j<size && kindCount<maxKind) {
            do {
                if (++counts[allNumbers[j].second]==1) {
                    kindCount++;
                }
                j++;
            } while (j<size && allNumbers[j-1].first==allNumbers[j].first);
        }
        
        if (kindCount<maxKind) {
            break;
        }
        
        //尾部跟进，破坏条件，最后得到的i是第一个破坏条件的，也代表选i-1的值时是满足条件的最小长度
        while (kindCount>=maxKind) {
            do {
                if (--counts[allNumbers[i].second]==0) {
                    kindCount--;
                }
                i++;
            } while (allNumbers[i-1].first==allNumbers[i].first);
        }
        
        printf("(%d,%d):(%d,%d),%d\n",i-1,j-1,allNumbers[i-1].first,allNumbers[j-1].first, kindCount);
        //更新解
        int range = allNumbers[j-1].first-allNumbers[i-1].first;
        if (range<minRange) {
            minRange = range;
            minStart = allNumbers[i-1].first;
        }
    }
    
    return {minStart, minRange+minStart};
}

vector<vector<string>> findDuplicate(vector<string> &paths) {
    //为了在输出结果的时候，不需要从files里把vector拷贝到result,在集合里路径有两个之后，就直接把数据写入到result. files的value的状态变化如下：
    //初始：没有值；找到一个文件，这个文件的路径；找到两个文件及以上，result里存储文件路径集合的索引
    //最后的状态里，通过索引可以把路径直接存入到result里
    unordered_map<string, string> files;
    vector<vector<string>> result;
    
    for (auto &str:paths){
        int start = 0, idx = 0;
        int bracket1=0,bracket2=0;
        string dir;
        for (auto &c:str){
            if (c==' ') {
                if (start==0) {
                    dir = str.substr(start, idx-start)+"/";
                }else{
                    auto &p = files[str.substr(bracket1+1, bracket2-bracket1-1)];
                    if (p.empty()) {
                        p = dir+str.substr(start, bracket1-start);
                    }else if(p[0]=='r'){
                        result.push_back({p, dir+str.substr(start, bracket1-start)});
                        p = to_string(result.size()-1);
                    }else{
                        result[extractNumber(p)].push_back(dir+str.substr(start, bracket1-start));
                    }
                }
                
                start = idx+1;
            }else if (c=='('){
                bracket1 = idx;
            }else if (c==')'){
                bracket2 = idx;
            }
            idx++;
        }
        
        auto &p = files[str.substr(bracket1+1, bracket2-bracket1-1)];
        if (p.empty()) {
            p = dir+str.substr(start, bracket1-start);
        }else if(p[0]=='r'){
            result.push_back({p, dir+str.substr(start, bracket1-start)});
            p = to_string(result.size()-1);
        }else{
            result[extractNumber(p)].push_back(dir+str.substr(start, bracket1-start));
        }
    }
    
    return result;
}

int main(int argc, const char * argv[]) {
    uint64_t start = mach_absolute_time();
    string path = "/Users/apple/Downloads/9 (2).in";
    
    
    
    vector<string> paths = {"root/qgjazhtliq/djmevsktisuvd/acsuolhnermqzok/mkts/ibrdqxawjgut/emb wl.txt(odumfqzwczehoyk) z.txt(gojsklotgchjzfm) txtoyg.txt(gojsklotgchjzfm) eupidhefx.txt(ahlsazuzrsf) rekzkaifwp.txt(yfxaymkefaofowqjpgaceffkjsehtmqkgy) l.txt(odumfqzwczehoyk) bqmhpxumxlbe.txt(yfxaymkefaofowqjpgaceffkjsehtmqkgy) qoqgiauqbayuc.txt(odumfqzwczehoyk) mpstemqlxy.txt(ahlsazuzrsf)"};
    findDuplicate(paths);
    
    
    uint64_t duration = mach_absolute_time() - start;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
    printf("exe time: %.1f ms\n",time);
}
