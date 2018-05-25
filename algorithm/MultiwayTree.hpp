//
//  MultiwayTree.hpp
//  algorithm
//
//  Created by shiwei on 2018/5/15.
//

#ifndef MultiwayTree_hpp
#define MultiwayTree_hpp

#include <stdio.h>
#include <stack>

using namespace std;

namespace TFDataStruct {
    template<class T>
    class MultiwayTreeNode {
        
    public:
        T val;
        vector<MultiwayTreeNode *> childern;
        
        int mark = 0;
        int select = 0;
        bool isFile = false;
        
        MultiwayTreeNode(T val):val(val){};
    };
}

using namespace TFDataStruct;

void calculateMaxLen(MultiwayTreeNode<string> *root){
    
    if (root->childern.empty()) {
        root->mark = root->isFile ? (int)(root->val.length()) : 0;
        return;
    }
    
    int maxLen = 0;
    for (auto &child : root->childern){
        calculateMaxLen(child);
        maxLen = max(maxLen, child->mark);
    }
    
    //必须要文件，文件夹不行
    if (maxLen != 0) {
        root->mark = maxLen+(int)(root->val.length())+1;
    }
}

//643. Longest Absolute File Path
/*
 坑总结：
 1. \n\t 这个是在字符串内就长这样，输入的时候实际是"\\n\\t"，导致的一个问题是，不能直接得到字符\n，它在输入里是\和n拆开的
 2. 找的最长路径只针对文件，文件夹不行，所以还要判断是否文件，不能文件或子路径里没有文件，返回长度都是0
 3. 顶级文件夹会有多个，所以不是找到一个顶级文件加就认为是root,而是要自建一个顶级文件夹，这样就成了单个树了，否则是多棵树，不好处理。
 4. 注意字符串遍历结束的时候，要处理最后一段文件，它在循环里没有。然后它是可能存在于任何层级的，并不总是在当前的下一级。
 */
int lengthLongestPath(string &input) {
    if (input.empty()) {
        return 0;
    }
    MultiwayTreeNode<string> *root = new MultiwayTreeNode<string>("");
    
    stack<MultiwayTreeNode<string> *> path;
    path.push(root);
    
    string cur = "";
    bool special = false;  //表示转移当前字符
    int layer = 1;
    bool isFile = false;
    for (char &c : input) {
        
//        printf("%c ",c);
        
        if (c == '\\') {  //开始计算下一个文件的层级
            special = true;
        }else{
            if (special) {
                if (c == 'n') {
                    while (path.size() > layer) {
                        path.pop();
                    }
                    
                    MultiwayTreeNode<string> *newNode = new MultiwayTreeNode<string>(cur);
                    newNode->isFile = isFile;
                    
                    path.top()->childern.push_back(newNode);
                    path.push(newNode);
                    
                    layer = 1;
                    cur = "";
                    isFile = false;
                    
                }if (c == 't') {
                    layer++;
                }
                special = false;
            }else{
                if (c == '.') {
                    isFile = true;
                }
                cur.push_back(c);
            }
        }
    }
    
    if (!cur.empty()) {
        while (path.size() > layer) {
            path.pop();
        }
        MultiwayTreeNode<string> *newNode = new MultiwayTreeNode<string>(cur);
        newNode->isFile = isFile;
        
        path.top()->childern.push_back(newNode);
    }
    
    calculateMaxLen(root);
    
    return max(root->mark-1, 0);
}

#endif /* MultiwayTree_hpp */
