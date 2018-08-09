//
//  LFUCache.hpp
//  algorithm
//
//  Created by shiwei on 2018/8/8.
//

#ifndef LFUCache_hpp
#define LFUCache_hpp

#include <stdio.h>

//1. 插入时，如果满了，最后一个一定要被淘汰，而不看它和插入的数据比较
//2. 插入已存在的数，访问数也要+1
//3. 更新的数都要放到同次数的第一个，也就是同次数的数据之间是LRU的
//4. 使用双层链表可以加快
class LFUCache {
    struct RowNode;
    struct KeyNode{
        int key;
        int value;
        
        KeyNode *next = nullptr;
        KeyNode *pre = nullptr;
        RowNode *row = nullptr;
        
        KeyNode(int key, int value, RowNode *row):key(key),value(value),row(row){};
        
        bool operator==(const KeyNode &other) const{
            return this->key == other.key;
        }
    };
    
    struct RowNode{
        int time = 0;
        
        RowNode(int time):time(time){};
        
        RowNode *next = nullptr;
        RowNode *pre = nullptr;
        
        KeyNode *keyHead = nullptr;
        KeyNode *keyTail = nullptr;
    };
    
    RowNode *lastRow = nullptr;
    
    int capacity = 0;
    int storedSize = 0;
    unordered_map<int, KeyNode*> store;
    
    void bringForward(KeyNode *node){
        RowNode *preRow = node->row->pre;
        if (preRow == nullptr || preRow->time != node->row->time+1) {
            
            //插入新的行
            preRow = new RowNode(node->row->time+1);
            preRow->next = node->row;
            preRow->pre = node->row->pre;
            
            if (preRow->next) preRow->next->pre = preRow;
            if (preRow->pre) preRow->pre->next = preRow;
        }
        
        removeKeyNode(node);
        if (node->row->keyHead == nullptr && node->row == lastRow) {
            delete node->row;
            lastRow = preRow;
        }
        insertNodeToFront(node, preRow);
    }
    
    inline void insertNodeToFront(KeyNode *node, RowNode *row){
        if (row->keyHead == node) {
            return;
        }
        
        node->next = row->keyHead;
        node->pre = nullptr;
        
        if (row->keyHead) {
            row->keyHead->pre = node;
        }else{
            row->keyHead = row->keyTail = node;
        }
        
        row->keyHead = node;
        node->row = row;
    }
    
    inline void removeKeyNode(KeyNode *node){
        
        if(node->pre) node->pre->next = node->next;
        if(node->next) node->next->pre = node->pre;
        
        if (node->row->keyHead == node) {
            if (node->row->keyTail == node){
                node->row->keyTail = node->row->keyHead = nullptr;
            }else{
                node->row->keyHead = node->next;
            }
        }else if (node->row->keyTail == node){
            node->row->keyTail = node->pre;
        }
    }
    
public:
    
    LFUCache(int capacity) {
        this->capacity = capacity;
        lastRow = new RowNode(1);
    }
    
    void set(int key, int value) {
        
        if (store.find(key) != store.end()) {
            KeyNode *find = store[key];
            find->value = value;
            bringForward(find);
            return;
        }
        
        KeyNode *newNode = nullptr;
        if (storedSize == capacity) { //满了，解除最后一个，修改值作为新节点
            newNode = lastRow->keyTail;
            removeKeyNode(lastRow->keyTail);
            
            store.erase(newNode->key);
            newNode->key = key;
            newNode->value = value;
            
            if (lastRow->keyHead == nullptr) {
                lastRow->time = 1;
            }
        }else{
            newNode = new KeyNode(key, value, lastRow);
            storedSize++;
        }
        
        if (lastRow->time != 1) {
            auto newLast = new RowNode(1);
            newLast->pre = lastRow;
            lastRow->next = newLast;
            
            lastRow = newLast;
        }
        
        store[key] = newNode;
        insertNodeToFront(newNode, lastRow);
        
        show();
    }
    
    int get(int key) {
        if (store.find(key) == store.end()) {
            return -1;
        }
        
        KeyNode *find = store[key];
        bringForward(find);
        
        show();
        return find->value;
    }
    
    void show(){
        RowNode *cur = lastRow;
        while (cur) {
            printf("\n###%d\n",cur->time);
            KeyNode *keyN = cur->keyTail;
            while (keyN) {
                printf("[%d : %d] ",keyN->key, keyN->value);
                keyN = keyN->pre;
            }
            
            cur = cur->pre;
        }
        
        printf("\n***************\n");
    }
};

#endif /* LFUCache_hpp */
