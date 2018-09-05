//
//  DoubleLink.hpp
//  algorithm
//
//  Created by shiwei on 2018/8/31.
//

////双链表

#ifndef DoubleLink_hpp
#define DoubleLink_hpp

#include <stdio.h>
#include <assert.h>

namespace TFDataStruct {
    
    template<class T>
    struct DoubleLinkNode{
        T val;
        DoubleLinkNode *next = nullptr;
        DoubleLinkNode *pre = nullptr;
        
        DoubleLinkNode(T val){
            this->val = val;
        };
    };
    
    template<class T>
    class Stack{
        
    public:
        DoubleLinkNode<T> *head = nullptr;
        DoubleLinkNode<T> *tail = nullptr;
        
        bool empty(){
            return head == nullptr;
        }
        
        void push(T val){
            auto newNode = new DoubleLinkNode<T>(val);
            if (tail == nullptr) {
                head = tail = newNode;
            }else{
                tail->next = newNode;
                newNode->pre = tail;
                
                tail = newNode;
            }
        }
        
        T pop(){
            assert(head);
            
            auto oldNode = tail;
            T value = oldNode->val;
            
            tail = tail->pre;
            if (tail) {
                tail->next = nullptr;
            }else{
                head = nullptr;
            }
            
            delete oldNode;
            return value;
        }
        
        T top(){
            assert(head);
            return tail->val;
        }
        
        
    };
}

#endif /* DoubleLink_hpp */
