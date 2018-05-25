//
//  BinaryTree.hpp
//  algorithm
//
//  Created by shiwei on 2018/5/16.
//

#ifndef BinaryTree_hpp
#define BinaryTree_hpp

#include <stdio.h>
#include <vector>
#include <stdlib.h>

using namespace std;

class TreeNode{
    
    
    
public:
    int val;
    TreeNode *left, *right;
    TreeNode(int val){
        this->val = val;
        this->left = this->right = NULL;
    }
    
    static TreeNode *createWithArray(vector<string> &nodeChars){
        if (nodeChars.empty() || nodeChars.front().front() == '#') {
            return nullptr;
        }
        
        TreeNode *root = new TreeNode(stoi(nodeChars.front()));
        vector<TreeNode *> *plane = new vector<TreeNode *>();
        plane->push_back(root);
        
        int index = 1;
        while (!plane->empty() && index < nodeChars.size()) {
            vector<TreeNode *> *nextPlane = new vector<TreeNode *>();
            for (auto node : *plane){
                
                if (index >= nodeChars.size()) {
                    break;
                }
                if (nodeChars[index].front() == '#') {
                    node->left = nullptr;
                }else{
                    node->left = new TreeNode(stoi(nodeChars[index]));
                    nextPlane->push_back(node->left);
                }
                index++;
                
                if (index >= nodeChars.size()) {
                    break;
                }
                if (nodeChars[index].front() == '#') {
                    node->right = nullptr;
                }else{
                    node->right = new TreeNode(stoi(nodeChars[index]));
                    nextPlane->push_back(node->right);
                }
                index++;
            }
            
            free(plane);
            plane = nextPlane;
        }
        
        free(plane);
        return root;
    }
    
    static vector<string> showTree(TreeNode *root){
        if (root == nullptr) {
            return {};
        }
        
        vector<string> result;
        
        vector<TreeNode *> *plane = new vector<TreeNode *>();
        plane->push_back(root);
        
        result.push_back(to_string(root->val));
        
        while (1) {
            
            vector<TreeNode *> *nextPlane = new vector<TreeNode *>();
            for (auto node : *plane){
                if (node->left) {
                    nextPlane->push_back(node->left);
                    result.push_back(to_string(node->left->val));
                }else{
                    result.push_back("#");
                }
                if (node->right) {
                    nextPlane->push_back(node->right);
                    result.push_back(to_string(node->right->val));
                }else{
                    result.push_back("#");
                }
            }
            
            if (nextPlane->empty()) {
                free(nextPlane);
                break;
            }
            free(plane);
            plane = nextPlane;
        }
        
        while (result.back().front() == '#') {
            result.pop_back();
        }
        free(plane);
        
        return result;
    }
    
    inline TreeNode *mostLeft(TreeNode *node){
        while (node->left) {
            node = node->left;
        }
        
        return node;
    }
    
    inline TreeNode *mostRight(TreeNode *node){
        while (node->right) {
            node = node->right;
        }
        
        return node;
    }
    
    inline static void swap(TreeNode *n1, TreeNode *n2){
        auto temp = n1->val;
        n1->val = n2->val;
        n2->val = temp;
    }
    
    static void inorderList(vector<TreeNode *> &list, TreeNode *node){
        if (node->left) {
            inorderList(list, node->left);
        }
        list.push_back(node);
        if (node->right) {
            inorderList(list, node->right);
        }
    }
};

TreeNode * trimBST(TreeNode * root, int minimum, int maximum) {
    bool left = true;
    TreeNode *cur = root;
    
    TreeNode superRoot(0); //建一个假的节点，用来做根节点的父节点,因为root也可能被切掉
    superRoot.left = root;
    superRoot.right = root;
    TreeNode *last = &superRoot;
    
    while (cur) {
        
        if (left) {
            if (minimum < cur->val) {
                last = cur;
                cur = cur->left;
            }else if (minimum == cur->val){
                cur->left = nullptr;
                break;
            }else{
                left = false;
                cur = cur->right;
                last->left = nullptr;  //当前节点是要被丢弃的
            }
        }else{
            if (minimum > cur->val) {
                cur = cur->right;
            }else{
                last->left = cur;
                
                if (minimum == cur->val){
                    cur->left = nullptr;
                    break;
                }
                left = true;
                last = cur;
                cur = cur->left;
            }
        }
    }
    
    
    bool right = true;
    cur = root;
    last = &superRoot;
    
    while (cur) {
        
        if (right) {
            if (maximum > cur->val) {
                last = cur;
                cur = cur->right;
            }else if (minimum == cur->val){
                cur->right = nullptr;
                break;
            }else{
                right = false;
                cur = cur->left;
                last->right = nullptr;  //当前节点是要被丢弃的
            }
        }else{
            if (maximum < cur->val) {
                cur = cur->left;
            }else{
                last->right = cur;
                
                if (minimum == cur->val){
                    cur->right = nullptr;
                    break;
                }
                right = true;
                last = cur;
                cur = cur->right;
            }
        }
    }
    
    if (superRoot.left == root) {
        if (superRoot.right == root) {
            return root;
        }else{
            return superRoot.right;
        }
    }else{
        return superRoot.left;
    }
}

#endif /* BinaryTree_hpp */
