//
//  MinStack.hpp
//  algorithm
//
//  Created by shiwei on 18/2/10.
//
//

#ifndef MinStack_hpp
#define MinStack_hpp

#include <stdio.h>

/** 栈具有进出的路线一致的性质，增加3个元素，然后减去3个，会回到之前一模一样的状态。所以对于具有n个元素的栈，它仅有n+1种不同的状态，并且可以用最后一个元素来做每种状态的位移标识，那么就可以把和状态对应的属性保存在最后一个节点里。这里就是把最小值保存在最后一个节点里。 */

class MinStack {
    
    struct Node{
        Node *next = nullptr;
        Node *pre = nullptr;
        Node *min = nullptr;
        
        int val;
        
        Node(int val):val(val){};
    };
    
    Node *head = nullptr;
    Node *tail = nullptr;
    
public:
    MinStack() {
        // do intialization if necessary
    }
    
    /*
     * @param number: An integer
     * @return: nothing
     */
    void push(int number) {
        
        Node *newNode = new Node(number);
        
        if (head == nullptr) {
            newNode->min = newNode;
            tail = head = newNode;
        }else{
            tail->next = newNode;
            newNode->pre = tail;
            
            if (number < tail->min->val) {
                newNode->min = newNode;
            }else{
                newNode->min = tail->min;
            }
            
            tail = newNode;
        }
    }
    
    /*
     * @return: An integer
     */
    int pop() {
        if (tail == nullptr) {
            throw (string)"stack is empty!";
            return 0;
        }
        
        int val = tail->val;
        
        Node *preTail = tail;
        tail = tail->pre;
        if (tail) {
            tail->next = nullptr;
        }else{
            head = nullptr;
        }
        free(preTail);
        
        return val;
    }
    
    /*
     * @return: An integer
     */
    int min() {
        if (tail == nullptr) {
            throw (string)"stack is empty!";;
        }
        
        return tail->min->val;
    }
    
    friend ostream& operator<<(ostream& os, MinStack& t){
        Node *cur = t.head;
        while (cur != nullptr) {
            os<<cur->val<<" ";
            cur = cur->next;
        }
        
        return os;
    }
};

#endif /* MinStack_hpp */
