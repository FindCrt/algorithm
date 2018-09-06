//
//  LRUCache.hpp
//  algorithm
//
//  Created by shiwei on 2018/9/5.
//

#ifndef LRUCache_h
#define LRUCache_h

#include <stdio.h>
#include <unordered_map>
#include <assert.h>

class LRUCache {
    struct CacheNode{
        int key;
        int value;
        
        CacheNode *pre = nullptr;
        CacheNode *next = nullptr;
        
        CacheNode(int key = 0, int value = 0):key(key),value(value){};
    };
    
    CacheNode *head = new CacheNode();
    CacheNode *tail = new CacheNode();
    
    int capacity;
    int size = 0;
    
    unordered_map<int,CacheNode *> store;
    
    inline void insertHead(CacheNode *node){
        node->next = head;
        node->pre = head->pre;
        
        node->pre->next = node;
        node->next->pre = node;
    }
    
    inline void unbind(CacheNode *node){
        node->next->pre = node->pre;
        node->pre->next = node->next;
    }
    
    /** 把节点拉到最前面 */
    inline void forword(CacheNode *node){
        if (node->next == head) { //已经是最前面了
            return;
        }
        unbind(node);
        insertHead(node);
    }
    
    /** 扔掉最后一个 */
    inline void dropTail(){
        auto drop = tail->next;
        unbind(drop);
        store.erase(drop->key);
        size--;
        delete drop;
    }
    
    friend ostream& operator<<(ostream& os, LRUCache &cache){
        auto cur = cache.tail->next;
        while (cur != cache.head) {
            os<<cur->key<<"->";
            cur = cur->next;
        }
        
        return os;
    }
    
public:
    
    LRUCache(int capacity) {
        assert(capacity); //容量等于0没法玩
        this->capacity = capacity;
        
        head->pre = tail;
        tail->next = head;
    }

    int get(int key) {
        if (store.find(key) == store.end()) {
            return -1;
        }else{
            auto find = store[key];
            forword(find);
            return find->value;
        }
    }
    
    void set(int key, int value) {
        if (store.find(key) == store.end()) {
            if (size == capacity) {
                dropTail();
            }
            
            auto node = new CacheNode(key, value);
            store[key] = node;
            insertHead(node);
            size++;
        }else{
            auto find = store[key];
            find->value = value;
            forword(find);
        }
    }
};

#endif /* LRUCache_h */
