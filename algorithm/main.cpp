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

bool wordPattern(string &pattern, string &teststr) {
    teststr.push_back(' ');
    
    //1. 前缀树记录的是单词到字符的对应关系，每个节点唯一对应一个单词(即从跟到这个节点的路径拼起来的单词),所以用节点存储单词映射的字符；
    //2. markC2W[c]表示c这个字符映射的单词，又因为节点和单词的一一对应关系，所以把单词对应的节点存入
    TFDataStruct::Trie<char> trieW2C;
    TFDataStruct::Trie<char>::TrieNode* charNodes[26];
    memset(charNodes, 0, sizeof(charNodes));
    
    //3. 因为要一一对应的关系，1和2刚好是两个方向
    
    int start=-1, i = 0;
    int wordIdx = 0;
    for (auto &c : teststr){
        
        if (start<0) {
            if (c!=' ') {
                start = i;
            }
        }else{
            if (c==' ') {
                auto str = teststr.substr(start, i-start);
                auto node = trieW2C.insert(str);
                
                auto charNode = charNodes[pattern[wordIdx]-'a'];
                if (charNode == nullptr) {
                    if (node->relateData>0) {
                        return false;
                    }
                }else if (node != charNode) {
                    return false;
                }
                
                node->relateData = pattern[wordIdx];
                charNodes[pattern[wordIdx]-'a'] = node;
                
                start = -1;
                wordIdx++;
            }
        }
        
        i++;
    }
    
    return true;
}

void maxPalindromeLen(string &str, int*oddLens, int *evenLens){
    int size = (int)str.length();
    oddLens[0] = 1;
    
    //奇数长度回文求一遍
    int maxI=0, maxR=1; //maxR是已求回文部分，最右侧的边界，maxI是这个回文的中心
    for (int i = 1; i<size; i++) {
        int curLen = 1; //这个不是回文的总长度，而是一半，准确的说是左边或者右边的长度，即包含中心
        if (i<maxR) {
            int mirrorLen = oddLens[2*maxI-i];
            if (mirrorLen+i<maxR) {//i...mirrorLen+i-1,[至少一个间隔],maxR
                curLen = mirrorLen;
            }else{
                curLen = maxR-i;
                int j = maxR;
                while (j<size && (2*i-j)>=0 && str[j]==str[2*i-j]) {
                    j++;
                    curLen++;
                }
                if (j>maxR) {  //能到达更远的地方，更新
                    maxR = j;
                    maxI = i;
                }
            }
        }else{
            int j = i+1;
            while (j<size && (2*i-j)>=0 && str[j]==str[2*i-j]) {
                j++;
                curLen++;
            }
            if (j>maxR) {  //能到达更远的地方，更新
                maxR = j;
                maxI = i;
            }
        }
        
        oddLens[i] = curLen;
    }
    
    //偶数回文求一遍
    maxI=-1;
    maxR=0;
    for (int i = 0; i<size; i++) {
        int curLen = 0; //偶数时，是准确的一半，中心偏左，这里的长度不包含中心
        if (i<maxR) {
            int mirrorLen = 2*maxI-i>=0?evenLens[2*maxI-i]:0;
            if (mirrorLen+i<maxR-1) {//i...mirrorLen+i,[至少一个间隔],maxR
                curLen = mirrorLen;
            }else{
                curLen = maxR-i-1;
                int j = maxR;
                while (j<size && (2*i+1-j)>=0 && str[j]==str[2*i+1-j]) {
                    j++;
                    curLen++;
                }
                if (j>maxR) {  //能到达更远的地方，更新
                    maxR = j;
                    maxI = i;
                }
            }
        }else{
            int j = i+1;
            while (j<size && (2*i+1-j)>=0 && str[j]==str[2*i+1-j]) {
                j++;
                curLen++;
            }
            if (j>maxR) {  //能到达更远的地方，更新
                maxR = j;
                maxI = i;
            }
        }
        
        evenLens[i] = curLen;
    }
    
    for (int i = 0; i<size; i++) {
        oddLens[i] = oddLens[i]*2-1;
        evenLens[i] = evenLens[i]*2;
    }
}

void maxPalindromeLen(string &str, int*lens){
    int size = (int)str.length();
    
    //这样处理，每个位置的回文长度都是奇数，1221=>1#2#2#1,第一个2位置的偶数回文变成了中心#的奇数回文
    //并且处理后左侧的长度就是原回文总长度，也就是处理后总长/2==原总长；所以len存储总长/2。
    int oddLen = 2*size+1;
    string oddStr(oddLen, '#');
    for (int i = 0; i<size; i++) {
        oddStr[2*i+1]=str[i];
    }
    
    int oddLens[oddLen];
    memset(oddLens, 0, sizeof(oddLens));
    oddLens[0] = 1;
    
    int maxI=0, maxR=1; //maxR是已求回文部分，最右侧的边界，maxI是这个回文的中心
    for (int i = 1; i<oddLen; i++) {
        int curLen = 1; //这个不是回文的总长度，而是一半，准确的说是左边或者右边的长度，即包含中心
        if (i<maxR) {
            
            int mirrorLen = oddLens[2*maxI-i];
            if (mirrorLen+i<maxR) {//i...mirrorLen+i-1,[至少一个间隔],maxR
                curLen = mirrorLen;
            }else{
                curLen = maxR-i;
                int j = maxR;
                while (j<oddLen && (2*i-j)>=0 && oddStr[j]==oddStr[2*i-j]) {
                    j++;
                    curLen++;
                }
                if (oddStr[j-1]=='#'){
                    j--;
                    curLen--;
                }
                if (j>maxR) {  //能到达更远的地方，更新
                    maxR = j;
                    maxI = i;
                }
            }
        }else{
            int j = i+1;
            while (j<oddLen && oddStr[j]==oddStr[2*i-j]) {
                j++;
                curLen++;
            }
            if (oddStr[j-1]=='#'){
                j--;
                curLen--;
            }
            if (j>maxR) {  //能到达更远的地方，更新
                maxR = j;
                maxI = i;
            }
        }
        
        printf("%d ",curLen);
        oddLens[i] = curLen;
    }
    
    for (int i = 0; i<size; i++) {
        lens[i] = max(oddLens[2*i+1], oddLens[2*i+2]);
    }
    
    printf("\n");
}

inline string reverseSub(string &str, int len, int end){
    string sub(len, ' ');
    for (int i = 0; i<len; i++) {
        sub[i] = str[end-i];
    }
    return sub;
}

vector<vector<int>> palindromePairs(vector<string> &words) {
    unordered_map<string, short> exists;
    int idx = 0;
    for (auto &w:words){
        exists[w]=idx;
        idx++;
    }
    
    vector<vector<int>> result;
    
    idx = 0;
    for (auto &w:words){
        if (w.empty()) {
            continue;
        }
        int wLen = (int)w.length();
        int oddLens[wLen];
        int evenLens[wLen];
        maxPalindromeLen(w, oddLens, evenLens);
        
//        printArrayOneLine(oddLens, wLen);
//        printArrayOneLine(evenLens, wLen);
        
        for (int i = 0; i<wLen; i++) {
            int pLens[2] = {oddLens[i], evenLens[i]};
            for (int j = 0; j<2; j++) {
                int pLen = pLens[j];
                if (pLen == 0) continue;
                if ((pLen-1)/2==i) {
                    auto iter = exists.find(reverseSub(w, wLen-pLen, wLen-1));
                    if (iter!=exists.end()) {
                        result.push_back({iter->second, idx});
                        if (pLen==wLen) { //自身是回文，对方是空字符串的特殊情况，空字符串还可接左边
                            result.push_back({idx, iter->second});
                        }
                    }
                }else if (pLen/2+i==wLen-1){
                    auto iter = exists.find(reverseSub(w, wLen-pLen, wLen-pLen-1));
                    if (iter!=exists.end()) {
                        result.push_back({idx, iter->second});
                    }
                }
            }
        }
        
        //自身全部逆转,如果存在，则接左边或右边都是回文,为了不重复计算，只记接右边的
        auto totalIter = exists.find(reverseSub(w, wLen, wLen-1));
        if (totalIter != exists.end() && totalIter->second != idx) {
            result.push_back({idx, totalIter->second});
        }
        
        idx++;
    }
    
    return result;
}

string longestPalindrome(string &s) {
    if (s.empty()) {
        return "";
    }
    int oddLens[s.length()];
    int evenLens[s.length()];
    maxPalindromeLen(s, oddLens, evenLens);
    printArrayOneLine(oddLens, s.length());
    printArrayOneLine(evenLens, s.length());
    int maxIdx = 0, maxLen = INT_MIN;
    for (int i = 0; i<s.length(); i++) {
        int ml = max(oddLens[i], evenLens[i]);
        if (ml>maxLen) {
            maxLen = ml;
            maxIdx = i;
        }
    }
    
    return s.substr(maxIdx-(maxLen-1)/2, maxLen);
}

struct IdxNode{
    uint8_t val;
    uint8_t next;
};

vector<int> anagramMappings(vector<int> &A, vector<int> &B) {
    uint8_t size = (uint8_t)A.size();
    unordered_map<int, vector<int>> idxB;
    
    for (uint8_t i = 0; i<size; i++) {
        idxB[B[i]].push_back(i);
    }
    
    vector<int> result;
    for (int i = 0; i<size; i++) {
        auto &idxes = idxB[A[i]];
        result.push_back(idxes.back());
        idxes.pop_back();
    }
    
    return result;
}

int subarraySumEqualsK(vector<int> &nums, int k) {
    int size = (int)nums.size();
    unordered_map<int, vector<int>> exist;
    
    int sum = 0;
    for (int i = 0; i<size; i++) {
        sum += nums[i];
        printf("%d ",sum);
        exist[sum].push_back(i);
    }
    printf("\n");
    
    auto i = exist.find(k);
    int result = i == exist.end()?0:i->second.size();
    for (auto &c : exist){
        auto iter = exist.find(c.first+k);
        if (iter != exist.end()) {
            for (auto &idx1 : c.second){
                for (auto &idx2 : iter->second){
                    if (idx2>idx1) {
                        printf("(%d, %d),%d-%d=%d\n",idx1,idx2,iter->first,c.first,k);
                    }
                    result += idx2>idx1?1:0;
                }
            }
        }
    }
    
    return result;
}

bool isSentenceSimilarity(vector<string> &words1, vector<string> &words2, vector<vector<string>> &pairs) {

    if (words1.size() != words2.size()) {
        return false;
    }
    
    unordered_map<string, bool> relations;
    
    for (auto &p : pairs) {
        relations[p.front()+p.back()] = true;
        relations[p.back()+p.front()] = true;
    }
    
    for (int i = 0; i<words1.size(); i++) {
        if (words1[i].compare(words2[i]) == 0) {
            continue;
        }
        if (relations.find(words1[i]+words2[i]) == relations.end()) {
            return false;
        }
    }
    
    return true;
}

#define add(i) ts.add(i);
#define find(i) {printf("%s\n",ts.find(i)?"true":"false");}
int main(int argc, const char * argv[]) {
    uint64_t start = mach_absolute_time();
    
    vector<string> word1 = {"great","acting","skills"};
    vector<string> word2 = {"fine","drama","talent"};
    vector<vector<string>> pairs = {{"great","fine"},{"drama","acting"},{"skills","talent"}};
    isSentenceSimilarity(word1, word2, pairs);
    
    uint64_t duration = mach_absolute_time() - start;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
    printf("exe time: %.1f ms\n",time);
}
