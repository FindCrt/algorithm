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
            uint32_t count = 0;
            T relateData;  //做额外的数据处理
            TrieNode *parent = nullptr;
            unordered_map<char, TrieNode*> childern;
        };
        
        typedef function<void(TrieNode *node, string &word, int idx)> TrieNodeVisitor;
        typedef function<void(TrieNode *node)> TrieNodeVisitor2;
    private:
    public:
        TrieNode root;
        
        Trie(){};
        Trie(vector<string> &words){
            for (auto &w : words){
                insert(w, nullptr);
            }
        }
        
        void iterateNodes(TrieNodeVisitor2 visitor){
            stack<TrieNode*> path;
            path.push(&root);
            
            while (!path.empty()) {
                auto top = path.top();
                path.pop();
                
                visitor(top);
                
                for (auto &p:top->childern){
                    if (p.second) path.push(p.second);
                }
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
                if (visitor) visitor(node, word, idx);
                
                idx++;
            }
            node->count++;
            
            return node;
        }
        
        inline bool exist(string &word){
            return count(word)>0;
        }
        
        inline int count(string &word){
            return count(word.begin(), word.end());
        }
        
        inline int count(const string::iterator &start, const string::iterator &end){
            return find(start, end)->count;
        }
        
        TrieNode *find(const string::iterator &start, const string::iterator &end){
            TrieNode *node = &root;
            auto iter = start;
            while (iter!=end) {
                auto &next = node->childern[*iter];
                if (next == nullptr) {
                    return 0;
                }
                node = next;
                iter++;
            }
            
            return node;
        }
    };
}

#endif /* Trie_hpp */
