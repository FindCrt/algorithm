//
//  AVL_tree.hpp
//  algorithm
//
//  Created by shiwei on 2019/1/17.
//

#ifndef AVL_tree_hpp
#define AVL_tree_hpp

#include <stdio.h>
#include <vector>

//实际使用应该是key-value的组合，这里只做了key的存储。
namespace TFDataStruct {
    
    template<class T>
    class AVL_Tree {
        
    public:
        struct TreeNode{
            T val;
            int balance;
            TreeNode *parent;
            TreeNode *left;
            TreeNode *right;
            
            TreeNode(T &val, TreeNode *parent):val(val),parent(parent){
                left = nullptr;
                right = nullptr;
                balance = 0;
            }
        };
        typedef TreeNode NodeType;
    private:
        
        NodeType *root = nullptr;
        
        //如果有对应值的节点，返回节点，否则返回这个值该连接的叶节点；exist判断是否存在
        NodeType *findNode(T &val, bool *exist){
            NodeType *cur = root, *last = nullptr;
            while (cur) {
                last = cur;
                if (cur->val == val) {
                    *exist = true;
                    return cur;
                }else if (cur->val > val){
                    cur = cur->left;
                }else{
                    cur = cur->right;
                }
            }
            
            *exist = false;
            return last;
        }
        
        /*
         原本左右子树的高度分别是a、b，假设左侧变化，高度变化i,则左侧高度为a+i,整体高度的变化为：
         (max(a+i, b)+1) - (max(a,b)+1) = max(a+i-b,0)-max(a-b,0) = max(balance2,0)-max(balance1,0)
         balance2 = a+i-b = (a-b)+i = balance1+i;
         
         如果是右子树，则：
         (max(a, b+i)+1) - (max(a,b)+1) = (max(a-i-b,0)+b+i)-(max(a-b,0)+b) = max(balance2,0)-max(balance1,0)+i
         balance2 = a-i-b = (a-b)-i = balance1-i;
         */
        
        //从某个叶节点开始检查并重新调整回复平衡,delta是node为根的子树的高度变化
        void rebalance(NodeType *node, int delta){
            NodeType *parent = node->parent;
            while (parent) {
                if (parent->left == node) {
                    int balance2 = parent->balance+delta;
                    delta = max(balance2,0)-max(parent->balance, 0);
                }else{
                    int balance2 = parent->balance-delta;
                    delta = max(balance2,0)-max(parent->balance, 0)+delta;
                }
                
                
                
                node = parent;
                parent = node->parent;
            }
        }
        
    public:
        AVL_Tree(vector<T> &vals){
            for (auto &v : vals){
                append(v);
            }
        }
        
        NodeType *getRoot(){
            return root;
        }
        
        void append(T &val){
            if (root == nullptr) {
                root = new NodeType(val, nullptr);
                return;
            }
            
            bool exist = false;
            NodeType *node = findNode(val, &exist);
            
            if (exist) {
                return;
            }
            
            NodeType *newNode = new NodeType(val, node);
            if (val<node->val) {
                node->left = newNode;
            }else{
                node->right = newNode;
            }
            
            rebalance(newNode, true);
        }
        
        
    };
}


#endif /* AVL_tree_hpp */
