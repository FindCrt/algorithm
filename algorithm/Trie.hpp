//
//  Trie.hpp
//  algorithm
//
//  Created by shiwei on 2019/1/25.
//

#ifndef Trie_hpp
#define Trie_hpp

#include <stdio.h>
#include <vector>
#include <map>

using namespace std;

namespace TFDataStruct {
    template<class T>
    class Trie{
    public:
        struct TrieNode{
            char ch;
            //对应的字符串的个数
            uint32_t mark = 0;
            T relateData;  //做额外的数据处理
            TrieNode *parent = nullptr;
            unordered_map<char, TrieNode*> childern;
        };
        struct TrieNodeVisitor{
            typedef void (*VisitFunc)(TrieNode *node, string &word, int idx);
            VisitFunc visitF;
            TrieNodeVisitor(VisitFunc visitF):visitF(visitF){};
        };
    private:
    public:
        TrieNode root;
        
        Trie(){};
        Trie(vector<string> &words){
            for (auto &w : words){
                insert(w, nullptr);
            }
        }
        
        /** 插入一个元素，同时最后的节点 */
        TrieNode *insert(string &word, TrieNodeVisitor visitor = nullptr){
            TrieNode *node = &root;
            int idx = 0;
            while (idx<word.length()) {
                auto &next = node->childern[word[idx]];
                if (next == nullptr) {
                    next = new TrieNode();
                    next->ch = word[idx];
                    next->parent = node;
                }
                node = next;
                if (visitor.visitF) visitor.visitF(node, word, idx);
                
                idx++;
            }
            node->mark++;
            
            return node;
        }
        
        int count(string &word){
            TrieNode *node = &root;
            int idx = 0;
            while (idx<word.length()) {
                auto &next = node->childern[word[idx]];
                if (next == nullptr) {
                    return 0;
                }
                node = next;
                idx++;
            }
            
            return node->mark;
        }
        
        bool exist(string &word){
            return count(word)>0;
        }
    };
}

#endif /* Trie_hpp */
