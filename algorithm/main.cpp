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

#define add(i) ts.add(i);
#define find(i) {printf("%s\n",ts.find(i)?"true":"false");}
int main(int argc, const char * argv[]) {
    uint64_t start = mach_absolute_time();
    
    
    string str1 = "aa";
    string str2 = "bog bod";
    auto result = wordPattern(str1, str2);
    printBool(result)
    
    uint64_t duration = mach_absolute_time() - start;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
    printf("exe time: %.1f ms\n",time);
}
