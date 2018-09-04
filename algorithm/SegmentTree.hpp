//
//  SegmentTree.hpp
//  algorithm
//
//  Created by shiwei on 2018/8/31.
//

////线段树

#ifndef SegmentTree_hpp
#define SegmentTree_hpp

#include <stdio.h>
#include "DoubleLink.hpp"

namespace TFDataStruct {
    /******* 模板类的线段树 ********/
    
    template<class ValueType>
    class SegmentTreeNode {
    public:
        int start, end;
        short mark = 0;
        ValueType value;
        SegmentTreeNode *left, *right;

        SegmentTreeNode(int start, int end, ValueType value = 0) {
            this->start = start;
            this->end = end;
            this->value = value;
            this->left = this->right = NULL;
        }
    };

    template<class ValueType, ValueType(*mergeFunc) (ValueType &node1, ValueType &node2)>
    class SegmentTree{
        
        const static short nodeMark_origin = 0;
        const static short nodeMark_traverseLeft = 1;
        const static short nodeMark_traverseRight = 2;

        typedef SegmentTreeNode<ValueType> NodeType;
        SegmentTreeNode<ValueType> *root;

    public:

        static NodeType * build(int start, int end){
            if (start > end) {
                return nullptr;
            }
            auto root = new NodeType(start, end);
            auto cur = root;

            TFDataStruct::Stack<NodeType *> path;

            do {
                if (cur->start == cur->end) {
                    if (path.empty()) {
                        break;
                    }
                    cur = path.pop();
                }else{
                    int mid = (cur->start+cur->end)/2;
                    auto left = new NodeType(cur->start, mid);
                    auto right = new NodeType(mid+1, cur->end);

                    cur->left = left;
                    cur->right = right;

                    path.push(right);
                    cur = left;
                }

            } while (1);

            return root;
        }
        
        /** 数组里是每个叶节点的统计数据，即数据和区间一起给了 */
        static NodeType * build(vector<ValueType> &A) {
            if (A.empty()) {
                return nullptr;
            }
            return build(A, 0, (int)A.size()-1);
        }
        
        static NodeType * build(vector<ValueType> &A, int start, int end){
            if (start > end) {
                return nullptr;
            }
            auto root = new NodeType(start, end);
            auto cur = root;
            
            TFDataStruct::Stack<NodeType *> path;
            
            do {
                if (cur->mark == nodeMark_origin) {
                    if (cur->start == cur->end) {  //到了叶节点，回溯
                        cur->value = A[cur->start];
                        if (path.empty()) {
                            break;
                        }
                        cur = path.pop();
                    }else{
                        //一个新的节点，构建左右节点，并切换到左树
                        int mid = (cur->start+cur->end)/2;
                        auto left = new NodeType(cur->start, mid);
                        auto right = new NodeType(mid+1, cur->end);
                        
                        cur->left = left;
                        cur->right = right;
                        
                        path.push(cur);
                        cur->mark = nodeMark_traverseLeft; //遍历左侧树时
                        
                        cur = left;
                    }
                }else if (cur->mark == nodeMark_traverseLeft){
                    cur->mark = nodeMark_traverseRight;    //遍历右侧树时
                    path.push(cur);
                    cur = cur->right;
                }else if (cur->mark == nodeMark_traverseRight){  //左右树都结束,重新计算值并回溯
                    cur->value = mergeFunc(cur->left->value, cur->right->value);
                    cur->mark = nodeMark_origin;
                    if (path.empty()) {
                        break;
                    }
                    cur = path.pop();
                }
                
            } while (1);
            
            return root;
        }

        /** 给指定区间内的节点都加上给定值 */
        static void add(SegmentTreeNode<ValueType> *root, int start, int end, ValueType delta){
            if (start > end || start > root->end || end < root->start) {
                return;
            }
            
            auto cur = root;
            TFDataStruct::Stack<NodeType *> path;
            
            do {
                if (cur->mark == nodeMark_origin) {
                    if (cur->start == cur->end) {  //到了叶节点，回溯
                        cur->value += delta;
                        if (path.empty()) {
                            break;
                        }
                        cur = path.pop();
                    }else{
                        //继续判断左右区间
                        int mid = (cur->start+cur->end)/2;
                        path.push(cur);
                        
                        if (mid>=start) { //左区间和目标重叠
                            cur->mark = nodeMark_traverseLeft;
                            cur = cur->left;
                        }else if (mid<end){  //左侧无重叠且右侧有重叠
                            cur->mark = nodeMark_traverseRight;
                            cur = cur->right;
                        }
                    }
                }else if (cur->mark == nodeMark_traverseLeft){
                    cur->mark = nodeMark_traverseRight;
                    if (cur->right->start<=end) { //右侧区间不满足时，就遍历了
                        path.push(cur);
                        cur = cur->right;
                    }
                }else if (cur->mark == nodeMark_traverseRight){  //左右树都结束,重新计算值并回溯
                    cur->value = mergeFunc(cur->left->value, cur->right->value);
                    cur->mark = nodeMark_origin;
                    if (path.empty()) {
                        break;
                    }
                    cur = path.pop();
                }
                
            } while (1);
        }

        static void modify(SegmentTreeNode<ValueType> *root, int index, ValueType newValue){
            if (index<root->start || index > root->end) {
                return;
            }
            
            auto cur = root;
            TFDataStruct::Stack<NodeType *> path;
            
            do {
                if (cur->mark == nodeMark_origin) {
                    if (cur->start == cur->end) {  //到了叶节点，回溯
                        cur->value = newValue;
                        if (path.empty()) {
                            break;
                        }
                        cur = path.pop();
                    }else{
                        //继续判断左右区间
                        int mid = (cur->start+cur->end)/2;
                        path.push(cur);
                        
                        if (mid>=index) { //左区间和目标重叠
                            cur->mark = nodeMark_traverseLeft;
                            cur = cur->left;
                        }else if (mid<index){  //左侧无重叠且右侧有重叠
                            cur->mark = nodeMark_traverseRight;
                            cur = cur->right;
                        }
                    }
                }else if (cur->mark == nodeMark_traverseLeft){
                    //单个点，左右区间里只有一个区间有可能，左侧遍历了，右侧就不需要了
                    cur->mark = nodeMark_traverseRight;
                }else if (cur->mark == nodeMark_traverseRight){  //左右树都结束,重新计算值并回溯
                    cur->value = mergeFunc(cur->left->value, cur->right->value);
                    cur->mark = nodeMark_origin;
                    if (path.empty()) {
                        break;
                    }
                    cur = path.pop();
                }
                
            } while (1);
        }

        /** 查询区间的值，defaultValue为 不包含目标区间时返回的值 */
        static ValueType query(SegmentTreeNode<ValueType> *root, int start, int end, ValueType defaultValue = ValueType()){
            if (start > end || start > root->end || end < root->start) {
                return defaultValue;
            }
            
            ValueType result;
            bool unassigned = true; //result是否被赋值过，因为数据类型是不确定，无法赋初始值
            auto cur = root;
            TFDataStruct::Stack<NodeType *> path;
            
            do {
                if (cur->mark == nodeMark_origin) {
                    //包含在目标区间里，直接取值【之前所有的操作都是为了这一步】
                    if (start <= cur->start && cur->end <= end) {
                        if (unassigned) {
                            result = cur->value;
                            unassigned = false;
                        }else{
                            result = mergeFunc(result, cur->value);
                        }
                        if (path.empty()) {
                            break;
                        }
                        cur = path.pop();
                    }else{
                        //继续判断左右区间
                        int mid = (cur->start+cur->end)/2;
                        path.push(cur);
                        
                        if (mid>=start) { //左区间和目标重叠
                            cur->mark = nodeMark_traverseLeft;
                            cur = cur->left;
                        }else if (mid<end){  //左侧无重叠且右侧有重叠
                            cur->mark = nodeMark_traverseRight;
                            cur = cur->right;
                        }
                    }
                }else if (cur->mark == nodeMark_traverseLeft){
                    cur->mark = nodeMark_traverseRight;
                    if (cur->right->start<=end) { //右侧区间不满足时，就遍历了
                        path.push(cur);
                        cur = cur->right;
                    }
                }else if (cur->mark == nodeMark_traverseRight){  //左右树都结束,重新计算值并回溯
                    cur->mark = nodeMark_origin;
                    if (path.empty()) {
                        break;
                    }
                    cur = path.pop();
                }
                
            } while (1);
            
            return result;
        }
    };
    
    
    /***** 用数组方式实现，而不是使用指针关联子节点，跟堆一样 ******/
    /////   性能并无明显提升，写起来却麻烦的要死  /////
    
//    #define SegTreeLeft(index) (index<<1)
//    #define SegTreeRight(index) (index<<1|1)
//
//    template<class ValueType>
//    class SegmentTreeNode {
//    public:
//        int start, end;
//        ValueType value;
//
//        SegmentTreeNode(int start, int end, ValueType value = 0) {
//            this->start = start;
//            this->end = end;
//            this->value = value;
//        }
//    };
//
//    template<class ValueType, ValueType(*mergeFunc) (ValueType &node1, ValueType &node2)>
//    class SegmentTree{
//
//        typedef SegmentTreeNode<ValueType> *NodeType *;
//        SegmentTreeNode<ValueType> *root;
//
//
//        static void build(vector<ValueType> &A, vector<NodeType *> &datas, int index, int start, int end){
//
//            datas[index] = new SegmentTreeNode<ValueType>(start, end);
//            SegmentTreeNode<ValueType> *cur = datas[index];
//
//            if (start < end) {
//                int mid = (start+end)/2;
//                build(A, datas, SegTreeLeft(index), start, mid);
//                build(A, datas, SegTreeRight(index), mid+1, end);
//
//                cur->value = mergeFunc(datas[SegTreeLeft(index)]->value, datas[SegTreeRight(index)]->value);
//            }else{
//                cur->value = A[start];
//            }
//        }
//
//        static void build(SegmentTreeNode<ValueType> **datas, int index, int start, int end){
//
//            datas[index] = new SegmentTreeNode<ValueType>(start, end);
//            SegmentTreeNode<ValueType> *cur = datas[index];
//
//            if (start < end) {
//                int mid = (start+end)/2;
//                build(datas, SegTreeLeft(index), start, mid);
//                build(datas, SegTreeRight(index), mid+1, end);
//            }
//        }
//
//        /** 给指定区间内的节点都加上给定值 */
//        static void add(vector<NodeType *> &datas, int curIndex, int start, int end, ValueType delta){
//
//            SegmentTreeNode<ValueType> *node = datas[curIndex];
//            if (node->start == node->end && (node->start >= start && node->start <= end)) {
//                node->value += delta;
//            }
//
//            int mid = (node->start+node->end)/2;
//
//            if (mid >= start) {
//                add(datas, SegTreeLeft(curIndex), start, end, delta);
//            }
//            if (mid < end) {
//                add(datas, SegTreeRight(curIndex), start, end, delta);
//            }
//
//            node->value = mergeFunc(datas[SegTreeLeft(curIndex)]->value, datas[SegTreeRight(curIndex)]->value);
//        }
//
//        static void modify(vector<NodeType *> &datas, int curIndex, int index, ValueType newValue){
//
//            SegmentTreeNode<ValueType> *node = datas[curIndex];
//            int mid = (node->start+node->end)/2;
//
//            if (mid >= index) {
//                modify(datas, SegTreeLeft(curIndex), newValue);
//            }else{
//                modify(datas, SegTreeRight(curIndex), newValue);
//            }
//
//            node->value = mergeFunc(datas[SegTreeLeft(curIndex)]->value, datas[SegTreeRight(curIndex)]->value);
//        }
//
//        static ValueType query(vector<NodeType *> &datas, int curIndex, int start, int end){
//
//            SegmentTreeNode<ValueType> *node = datas[curIndex];
//            if (node->start == node->end && (node->start >= start && node->start <= end)) {
//                return node->value;
//            }
//
//            ValueType result = 0;
//            int mid = (node->start+node->end)/2;
//
//            if (mid >= start) {
//                result = query(datas, SegTreeLeft(curIndex), start, end);
//                if (mid < end) {
//                    auto rightValue = query(datas, SegTreeRight(curIndex), start, end);
//                    result = mergeFunc(result, rightValue);
//                }
//            }else if (mid < end){
//                result = query(datas, SegTreeRight(curIndex), start, end);
//            }
//
//            return result;
//        }
//
//    public:
//
//
//
//        /** 数组里是每个叶节点的统计数据，即数据和区间一起给了 */
//        static vector<NodeType *> build(vector<ValueType> &A) {
//
//            vector<NodeType *> datas(A.size()*2, nullptr);
//            build(A, datas, 0, 0, (int)A.size()-1);
//            return datas;
//        }
//
//        /** 只给定区间,把整个线段树建立起来，但是还没有值；这种适合没有直接的叶节点统计数据，数据是伴随着区间的，先建树，再通过线段树的区间修改方法，这样建树更快 */
//        static SegmentTreeNode<ValueType> * build(int start, int end) {
//            if (start > end) {
//                return;
//            }
//
//            SegmentTreeNode<ValueType> *datas[(end-start+1)*2];
//            build(datas, 0, 0, end-start);
//            return datas[0];
//        }
//
//        static void add(vector<NodeType *> &datas, int start, int end, ValueType delta){
//            add(datas, 0, start, end, delta);
//        }
//
//        static void modify(vector<NodeType *> &datas, int index, ValueType newValue){
//            modify(datas, 0, index, newValue);
//        }
//
//        static ValueType query(vector<NodeType *> &datas, int start, int end){
//            return query(datas, 0, start, end);
//        }
//    };
}

#endif /* SegmentTree_hpp */
