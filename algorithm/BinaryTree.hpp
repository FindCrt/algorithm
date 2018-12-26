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
#include <stack>
#include "CommonStructs.hpp"

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
    
    static TreeNode *createFromBracketString(string str){
        stack<TreeNode *> parents;
        TreeNode *root = nullptr;
        
        TreeNode *last = nullptr;
        bool left = false;
        for (char &c : str){
            if (c == ' ') {
                continue;
            }
            if (c == '(') {
                parents.push(last);
                left = true;
            }else if (c == ')'){
                parents.pop();
            }else if (c == ','){
                left = false;
            }else{
                TreeNode *node = new TreeNode(c-'0');
                if (parents.empty()) {
                    root = node;
                }else{
                    if (left){
                        parents.top()->left = node;
                    }else{
                        parents.top()->right = node;
                    }
                }
                last = node;
            }
        }
        
        return root;
    }
    
    //左节点时，输出"(",切换到右子树时，输出“,”,结束返回时输出“)”.所以3个状态都需要，只能是后续遍历了。
    //或者直接在栈里存储节点的状态
    static string showTreeWithBlacketString(TreeNode *root){
        if (root == nullptr) {
            return "";
        }
        
        string result = "";
        stack<pair<TreeNode *, short>> path;
        path.push({root, 1});
        
        while (!path.empty()) {
            auto &cur = path.top();
            
            if (cur.second == 1) { //is left node
                result += to_string(cur.first->val);
                
                cur.second++;
                if (cur.first->left) {
                    path.push({cur.first->left, 1});
                    result += "(";
                    continue;
                }
            }
            
            if (cur.second == 2){
                cur.second++;
                
                if (cur.first->right) {
                    path.push({cur.first->right, 1});
                    if (cur.first->left) {
                        result += ",";
                    }else{
                        result += "(,";
                    }
                    continue;
                }
            }
            
            if (cur.first->left || cur.first->right) {
                result += ")";
            }
            path.pop();
        }
        
        return result;
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
    
    /** 先序遍历是最方便的 */
    static TreeNode *findNode(TreeNode *root, int val){
        if (root == nullptr) {
            return nullptr;
        }
        
        stack<TreeNode *> path;
        path.push(root);
        
        while (!path.empty()) {
            
            auto cur = path.top();
            if (cur->val == val) {
                return cur;
            }
            path.pop();
            
            if (cur->right) {
                path.push(cur->right);
            }
            if (cur->left) {
                path.push(cur->left);
            }
        }
        
        return nullptr;
    }
    
    //先序遍历
    static void release(TreeNode *root){
        if (root == nullptr) {
            return;
        }
        
        stack<TreeNode *> path;
        path.push(root);
        
        while (!path.empty()) {
            
            auto cur = path.top();
            path.pop();
            
            if (cur->right) {
                path.push(cur->right);
            }
            if (cur->left) {
                path.push(cur->left);
            }
            
            delete cur;
        }
    }
    
    static int maxDepth(TreeNode *root) {
        if (root == nullptr) {
            return 0;
        }
        
        stack<pair<TreeNode *, int>> path;
        path.push({root, 1});
        
        int maxDeap = 0;
        
        while (!path.empty()) {
            
            auto cur = path.top();
            auto parent = cur.first;
            path.pop();
            
            int count = 0;
            if (parent->right) {
                path.push({parent->right, cur.second+1});
                count++;
            }
            if (parent->left) {
                path.push({parent->left, cur.second+1});
                count++;
            }
            
            if (count == 0 && cur.second > maxDeap) {
                maxDeap = cur.second;
            }
        }
        
        return maxDeap;
    }
    
    struct PathSum{
        TreeNode *node = nullptr;
        int sum = 0;
        int state = 0;
        
        PathSum(TreeNode *node, int sum, int state):node(node),sum(sum),state(state){};
    };
    
    //路径值的和为目标值
    static vector<vector<int>> binaryTreePathSum(TreeNode *root, int target) {
        
        vector<vector<int>> result;
        if (root == nullptr) {
            return result;
        }
        
        vector<int> valPath;
        
        stack<PathSum> path;
        path.push({root, 0, 0});
        
        while (!path.empty()) {
            
            auto &cur = path.top();
            auto &node = cur.node;
            
            //遍历左边
            if (cur.state == 0) {
                cur.sum += node->val;
                valPath.push_back(node->val);
                
                cur.state++;
                if (node->left) {
                    path.push({node->left, cur.sum, 0});
                    continue;
                }
                
            }
            
            //遍历右边
            if (cur.state == 1) {
                cur.state++;
                if (node->right) {
                    path.push({node->right, cur.sum, 0});
                    continue;
                }
            }
            
            //回溯
            if (cur.state == 2) {
                if (cur.sum == target &&
                    node->left == nullptr &&
                    node->right == nullptr) {
                    
                    result.push_back(valPath);
                }
                
                path.pop();
                valPath.pop_back();
            }
        }
        
        return result;
    }
    
    //翻转二叉树，前序遍历就可以完成
    static void invertBinaryTree(TreeNode *root) {
        if (root == nullptr) {
            return;
        }
        
        stack<TreeNode *>path;
        path.push(root);
        
        while (!path.empty()) {
            
            auto cur = path.top();
            
            TreeNode *left = cur->left;
            cur->left = cur->right;
            cur->right = left;
            
            path.pop();
            if (cur->left) {
                path.push(cur->left);
            }
            if (cur->right) {
                path.push(cur->right);
            }
        }
    }
    
    static int getNodeCount(TreeNode * root) {
        if (root == nullptr) {
            return 0;
        }
        
        stack<TreeNode *>path;
        path.push(root);
        
        int count = 0;
        
        while (!path.empty()) {
            
            count++;
            auto cur = path.top();
            path.pop();
            if (cur->left) {
                path.push(cur->left);
            }
            if (cur->right) {
                path.push(cur->right);
            }
        }
        
        return count;
    }
};

static TreeNode * trimBST(TreeNode * root, int minimum, int maximum) {
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
