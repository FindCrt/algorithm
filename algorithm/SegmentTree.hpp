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
    
    class SegmentTreeNode_x {
        public:
        int start, end, max, count;
        SegmentTreeNode_x *left, *right;
        SegmentTreeNode_x(int start, int end) {
            this->start = start;
            this->end = end;
            this->max = INT_MIN;
            this->left = this->right = NULL;
        }
        
        SegmentTreeNode_x(int start, int end, int max) {
            this->start = start;
            this->end = end;
            this->max = max;
            this->left = this->right = NULL;
        }
    };
    
    class SegmentTree_x{
        SegmentTreeNode_x *root;
        
    public:
        
        static SegmentTreeNode_x * build_recursive(int start, int end) {
            auto cur = new SegmentTreeNode_x(start, end);
            
            if (start < end) {
                int mid = (start+end)/2;
                cur->left = build_recursive(start, mid);
                cur->right = build_recursive(mid+1, end);
            }
            
            return cur;
        }
        
        static SegmentTreeNode_x * build(int start, int end){
            if (start > end) {
                return nullptr;
            }
            auto root = new SegmentTreeNode_x(start, end);
            auto cur = root;
            
            TFDataStruct::Stack<SegmentTreeNode_x *> path;
            
            do {
                if (cur->start == cur->end) {
                    if (path.empty()) {
                        break;
                    }
                    cur = path.pop();
                }else{
                    int mid = (cur->start+cur->end)/2;
                    auto left = new SegmentTreeNode_x(cur->start, mid);
                    auto right = new SegmentTreeNode_x(mid+1, cur->end);
                    
                    cur->left = left;
                    cur->right = right;
                    
                    path.push(right);
                    cur = left;
                }

            } while (1);
            
            return root;
        }
        
        /*** 值为区间最大值的相关方法 */
        
        static SegmentTreeNode_x * build_max(vector<int> &A, int start, int end){
            auto cur = new SegmentTreeNode_x(start, end);
            
            if (start < end) {
                int mid = (start+end)/2;
                cur->left = build_max(A, start, mid);
                cur->right = build_max(A, mid+1, end);
                
                cur->max = max(cur->left->max, cur->right->max);
            }else{
                cur->max = A[start];
            }
            
            return cur;
        }
        
        static SegmentTreeNode_x * build_max(vector<int> &A) {
            if (A.empty()) {
                return nullptr;
            }
            return build_max(A, 0, (int)A.size()-1);
        }
        
        static void query_max(SegmentTreeNode_x * root, int start, int end, int &maxVal) {
            if (root->start >= start && root->end <= end) {
                maxVal = max(maxVal,root->max);
            }else{
                int mid = (root->start+root->end)/2;
                if (mid>=start) {
                    query_max(root->left, start, end, maxVal);
                }
                if (mid<end) {
                    query_max(root->right, start, end, maxVal);
                }
            }
        }
        
        static int query_max(SegmentTreeNode_x * root, int start, int end) {
            int maxVal = INT_MIN;
            query_max(root, start, end, maxVal);
            return maxVal;
        }
        
        
        /** 用于计数的线段树 */
        
        static SegmentTreeNode_x * build_count(vector<int> &A) {
            if (A.empty()) {
                return nullptr;
            }
            return build_count(A, 0, (int)A.size()-1);
        }
        
        static SegmentTreeNode_x * build_count(vector<int> &A, int start, int end){
            auto cur = new SegmentTreeNode_x(start, end);
            
            if (start < end) {
                int mid = (start+end)/2;
                cur->left = build_count(A, start, mid);
                cur->right = build_count(A, mid+1, end);
                
                cur->count = cur->left->count + cur->right->count;
            }else{
                cur->count = A[start];
            }
            
            return cur;
        }
        
        static void query_count(SegmentTreeNode_x * root, int start, int end, int &count){
            if (root->start >= start && root->end <= end) {
                count += root->count;
                cout<<"["<<root->start<<","<<root->end<<"] "<<root->count<<"-> "<<count<<endl;
            }else if (root->left < root->right){
                
                int mid = (root->start+root->end)/2;
                
                if (mid>=start) {
                    query_count(root->left, start, end, count);
                }
                if (mid<end) {
                    query_count(root->right, start, end, count);
                }
            }
        }
        
        static int query_count(SegmentTreeNode_x * root, int start, int end) {
            if (root == nullptr) {
                return 0;
            }
            
            int count = 0;
            query_count(root, start, end, count);
            return count;
        }
        
        /** 线段树修改 */
        
        static void modify_max(SegmentTreeNode_x * root, int index, int value) {
            if (index < root->start || index > root->end) {
                return;
            }
            
            TFDataStruct::Stack<SegmentTreeNode_x *> path;
            
            SegmentTreeNode_x *cur = root;
            while (cur) {
                
                if (cur->start == cur->end) {
                    cur->max = value;
                    break;
                }
                path.push(cur);
                
                if (index > cur->left->end) {
                    cur = cur->right;
                }else{
                    cur = cur->left;
                }
            }
            
            while (!path.empty()) {
                auto node = path.pop();
                node->max = max(node->left->max, node->right->max);
            }
        }
        
        
    };
    
    
    /******* 模板类的线段树 ********/
    
    template<class ValueType>
    class SegmentTreeNode {
    public:
        int start, end;
        ValueType value;
        SegmentTreeNode *left, *right;

        SegmentTreeNode(int start, int end, ValueType value) {
            this->start = start;
            this->end = end;
            this->value = value;
            this->left = this->right = NULL;
        }
    };

    template<class ValueType, ValueType(*mergeFunc) (ValueType &node1, ValueType &node2)>
    class SegmentTree{

        SegmentTreeNode<ValueType> *root;

        static SegmentTreeNode<ValueType> * build(vector<ValueType> &A, int start, int end){
            auto cur = new SegmentTreeNode<ValueType>(start, end, 0);

            if (start < end) {
                int mid = (start+end)/2;
                cur->left = build(A, start, mid);
                cur->right = build(A, mid+1, end);

                cur->value = mergeFunc(cur->left->value, cur->right->value);
            }else{
                cur->value = A[start];
            }

            return cur;
        }

    public:

        /** 数组里是每个叶节点的统计数据，即数据和区间一起给了 */
        static SegmentTreeNode<ValueType> * build(vector<ValueType> &A) {
            if (A.empty()) {
                return nullptr;
            }
            return build(A, 0, (int)A.size()-1);
        }

        /** 只给定区间,把整个线段树建立起来，但是还没有值；这种适合没有直接的叶节点统计数据，数据是伴随着区间的，先建树，再通过线段树的区间修改方法，这样建树更快 */
        static SegmentTreeNode<ValueType> * build(int start, int end) {
            auto cur = new SegmentTreeNode<ValueType>(start, end, 0);

            if (start < end) {
                int mid = (start+end)/2;
                cur->left = build(start, mid);
                cur->right = build(mid+1, end);
            }

            return cur;
        }

        /** 给指定区间内的节点都加上给定值 */
        static void add(SegmentTreeNode<ValueType> *root, int start, int end, ValueType delta){
            if (root->start == root->end && (root->start >= start && root->start <= end)) {
                root->value += delta;
            }

            int mid = (root->start+root->end)/2;

            if (mid >= start) {
                add(root->left, start, end, delta);
            }
            if (mid < end) {
                add(root->right, start, end, delta);
            }

            root->value = mergeFunc(root->left->value, root->right->value);
        }

        static void modify(SegmentTreeNode<ValueType> *root, int index, ValueType newValue){
            int mid = (root->start+root->end)/2;

            if (mid >= index) {
                modify(root->left, index, newValue);
            }else{
                modify(root->right, index, newValue);
            }

            root->value = mergeFunc(root->left->value, root->right->value);
        }

        static ValueType query(SegmentTreeNode<ValueType> *root, int start, int end){

            if (root->start == root->end && (root->start >= start && root->start <= end)) {
                return root->value;
            }

            ValueType result = 0;
            int mid = (root->start+root->end)/2;

            if (mid >= start) {
                result = query(root->left, start, end);
                if (mid < end) {
                    auto rightValue = query(root->right, start, end);
                    result = mergeFunc(result, rightValue);
                }
            }else if (mid < end){
                result = query(root->right, start, end);
            }

            return result;
        }
    };
    
    
    /***** 用数组方式实现，而不是使用指针关联子节点，跟堆一样 ******/
    
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
//        typedef SegmentTreeNode<ValueType> *NodeType;
//        SegmentTreeNode<ValueType> *root;
//
//
//        static void build(vector<ValueType> &A, vector<NodeType> &datas, int index, int start, int end){
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
//        static void add(vector<NodeType> &datas, int curIndex, int start, int end, ValueType delta){
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
//        static void modify(vector<NodeType> &datas, int curIndex, int index, ValueType newValue){
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
//        static ValueType query(vector<NodeType> &datas, int curIndex, int start, int end){
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
//        static vector<NodeType> build(vector<ValueType> &A) {
//
//            vector<NodeType> datas(A.size()*2, nullptr);
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
//        static void add(vector<NodeType> &datas, int start, int end, ValueType delta){
//            add(datas, 0, start, end, delta);
//        }
//
//        static void modify(vector<NodeType> &datas, int index, ValueType newValue){
//            modify(datas, 0, index, newValue);
//        }
//
//        static ValueType query(vector<NodeType> &datas, int start, int end){
//            return query(datas, 0, start, end);
//        }
//    };
}

#endif /* SegmentTree_hpp */
