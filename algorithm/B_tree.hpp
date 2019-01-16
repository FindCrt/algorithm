//
//  B_tree.hpp
//  algorithm
//
//  Created by shiwei on 2019/1/15.
//

#ifndef B_tree_hpp
#define B_tree_hpp

#include <stdio.h>
#include <vector>

//B-树和B+树的实现

namespace TFDataStruct {
    
    template<class keyTp, class valTp>
    class B_Tree{
        
        struct B_TreeNode{
            B_TreeNode *parent;
            int size;
            vector<keyTp> keys;
            vector<valTp> vals;
            vector<B_TreeNode *> childern;
            
            B_TreeNode(){}
            
            B_TreeNode(keyTp &key, valTp &val){
                keys.push_back(key);
                vals.push_back(val);
                parent = nullptr;
                size = 2;
                childern = {nullptr, nullptr};
            }
        };
        
        typedef B_TreeNode NodeType;
        
        NodeType *find(keyTp &key, int *keyIdx, bool *exist){
            NodeType *cur = root;
            NodeType *last = nullptr;
            while (cur) {
                int idx = binaryFind(cur->keys, key, bFindNotFoundTypeGreater);
                *keyIdx = idx;
                if (idx < cur->size-1 && cur->keys[idx]==key) {
                    *exist = true;
                    return cur;
                }else{
                    //没找到使用后一个的索引，即key属于[idx-1, idx]之间，这个区域对应孩子索引就是idx
                    last = cur;
                    cur = cur->childern[idx];
                }
            }
            
            *exist = false;
            return last;
        }
        
        void divide(NodeType *node){
            //把关键字分成左边、中间、右边3部分，中间插入到父节点，左右分别分配到分裂后的两个节点里
            int leftCount = maxSize/2;  //左边关键字个数
            int rightCount = maxSize-leftCount-1;
            
            keyTp &midKey = node->keys[leftCount];
            int pKeyIdx = 0;
            if (node->parent == nullptr) {
                //如果父节点为空，代表当前已经是根了，这时需要新建一个节点作为根；
                //B树的扩张是从顶部向上的
                root = node->parent = new NodeType(); //根变了
                root->size = 1;
                root->childern.push_back(node);
            }else{
                pKeyIdx = binaryFind(node->parent->keys, midKey, bFindNotFoundTypeGreater);
            }
            
            //把中间关键字插入到父节点里，父节点也可能会继续分裂
            insertKey(midKey, node->vals[leftCount], node->parent, pKeyIdx);
            
            //新建一个节点作为分裂后的右节点
            NodeType *other = new NodeType();
            other->size = rightCount+1;
            other->parent = node->parent;
            other->keys.insert(other->keys.begin(), node->keys.begin()+leftCount+1, node->keys.end());
            other->vals.insert(other->vals.begin(), node->vals.begin()+leftCount+1, node->vals.end());
            for (int i = leftCount+1; i<node->size; i++){
                other->childern.push_back(node->childern[i]);
                if (node->childern[i]) node->childern[i]->parent = other;
            }
            node->parent->childern[pKeyIdx+1] = other;
            
            //原节点内容缩小，变成分裂后的左节点
            node->size = leftCount+1;
            node->keys.erase(node->keys.begin()+leftCount, node->keys.end());
            node->vals.erase(node->vals.begin()+leftCount, node->vals.end());
            node->childern.erase(node->childern.begin()+leftCount+1, node->childern.end());
            
            //超过最大子节点个数，分裂
            if (node->parent->size > maxSize) {
                divide(node->parent);
            }
        }
        
        //在指定节点的指定位置插入关键字
        void insertKey(keyTp &key, valTp &val, NodeType *node, int keyIdx){
            
            node->size++;
            node->keys.insert(node->keys.begin()+keyIdx, key);
            node->vals.insert(node->vals.begin()+keyIdx, val);
            node->childern.insert(node->childern.begin()+keyIdx+1, nullptr);
        }
        
        //把节点first和first+1位置的孩子融合
        void mergeChildern(NodeType *node, int first){
            NodeType *left = node->childern[first];
            NodeType *right = node->childern[first+1];
            left->size += right->size;
            
            //先添加分隔key
            left->keys.insert(left->keys.end(), node->keys[first]);
            left->vals.insert(left->vals.end(), node->vals[first]);
            
            //再把右侧节点的数据全部导入到左边
            left->keys.insert(left->keys.end(), right->keys.begin(),right->keys.end());
            left->vals.insert(left->vals.end(), right->vals.begin(),right->vals.end());
            for (auto &c : right->childern){
                if (c) c->parent = left;
                left->childern.push_back(c);
            }
            
            //把分隔key去掉
            node->keys.erase(node->keys.begin()+first);
            node->vals.erase(node->vals.begin()+first);
            node->childern.erase(node->childern.begin()+first+1);
            node->size--;
            
            adjustShort(node);
        }
        
        //调整不足
        void adjustShort(NodeType *node){
            
            if (node == root) {
                if (node->size == 1) { //根没有key了,根降到下一级
                    root = node->childern.front();
                    delete node;
                }
                return;
            }
            
            //删除之后会导致关键字少于最小值 上界(maxSize/2)
            int minSize = (maxSize-1)/2+1;
            if (node->size<minSize) {
                int pKeyIdx = binaryFind(node->parent->keys, node->keys.front(), bFindNotFoundTypeSmaller);
                //左边有多余可借,左边最后一个换到父节点里，父节点的key换到当前节点里做第一个
                if (pKeyIdx>=0) {
                    NodeType *left = node->parent->childern[pKeyIdx];
                    if (left->size > minSize) {
                        node->size++;
                        left->size--;
                        
                        node->keys.insert(node->keys.begin(), node->parent->keys[pKeyIdx]);
                        node->parent->keys[pKeyIdx]=left->keys.back();
                        left->keys.erase(left->keys.end()-1);
                        
                        node->vals.insert(node->vals.begin(), node->parent->vals[pKeyIdx]);
                        node->parent->vals[pKeyIdx]=left->vals.back();
                        left->vals.erase(left->vals.end()-1);
                        
                        node->childern.push_back(nullptr);
                        left->childern.pop_back();
                        return;
                    }
                }
                
                //pKeyIdx:left; pKeyIdx+1:this node; pKeyIdx+2:right
                if (pKeyIdx+2 < node->parent->size) {
                    NodeType *right = node->parent->childern[pKeyIdx+2];
                    //右侧借一个关键字到左边
                    if (right->size>minSize) {
                        node->size++;
                        right->size--;
                        
                        node->keys.push_back(node->parent->keys[pKeyIdx+1]);
                        node->parent->keys[pKeyIdx+1]=right->keys.front();
                        right->keys.erase(right->keys.begin());
                        
                        node->vals.push_back(node->parent->vals[pKeyIdx+1]);
                        node->parent->vals[pKeyIdx+1]=right->vals.front();
                        right->vals.erase(right->vals.begin());
                        
                        node->childern.push_back(nullptr);
                        right->childern.erase(right->childern.begin());
                        return;
                    }
                }
                
                //左边和右边的key都不足，则把当前节点和左边或右边融合
                if (pKeyIdx>=0) {
                    mergeChildern(node->parent, pKeyIdx);
                }else{
                    mergeChildern(node->parent, pKeyIdx+1);
                }
            }
        }
        
        NodeType* succeedNode(NodeType *node, int keyIdx){
            NodeType *cur = node->childern[keyIdx+1];
            NodeType *last = nullptr;
            while (cur) {
                last = cur;
                cur = cur->childern.front();
            }
            
            return last;
        }
        
        void eraseKey(NodeType *node, int keyIdx){
            
            if (node->childern.front() != nullptr) {
                //中间节点，使用右侧最小的关键代替，转嫁为叶节点的删除
                NodeType *replace = succeedNode(node, keyIdx);
                node->keys[keyIdx] = replace->keys.front();
                node->vals[keyIdx] = replace->vals.front();
                
                eraseKey(replace, 0);
                return;
            }
            
            //叶节点，直接删除
            node->keys.erase(node->keys.begin()+keyIdx);
            node->vals.erase(node->vals.begin()+keyIdx);
            node->childern.erase(node->childern.begin()+keyIdx);
            node->size--;
            
            adjustShort(node);
        }
        
    public:
        int maxSize = 3;
        NodeType *root = nullptr;
        
        B_Tree(vector<keyTp> &keys, int maxSize, valTp defaultVal){
            this->maxSize = maxSize;
            for (auto &key : keys){
                this->append(key, defaultVal);
            }
        }
        
        B_Tree(vector<keyTp> &keys, int maxSize, vector<valTp> &vals){
            this->maxSize = maxSize;
            for (int i = 0; i<keys.size(); i++){
                this->append(keys[i], vals[i]);
            }
        }
        
        /** 添加一个关键字 */
        void append(keyTp &key, valTp &val){
            if (root == nullptr) {
                root = new NodeType(key, val);
            }else{
                int keyIdx = 0;
                bool exist = false;
                NodeType *node = find(key, &keyIdx, &exist);
                //如果存在，只做值的修正
                if (exist) {
                    node->vals[keyIdx] = val;
                    return;
                }
                
                insertKey(key, val, node, keyIdx);
                //超过最大子节点个数，分裂
                if (node->size > maxSize) {
                    divide(node);
                }
            }
        }
        
        void erase(keyTp &key){
            int keyIdx = 0;
            bool exist = false;
            NodeType *node = find(key, &keyIdx, &exist);
            
            if (!exist) {
                return;
            }
            
            eraseKey(node, keyIdx);
        }
        
        //按层输出B树的关键字
        void show(){
            queue<pair<NodeType*, int>> nodeQ;
            nodeQ.push({root, 1});
            
            int curLevel = 1;
            while (!nodeQ.empty()) {
                NodeType *cur = nodeQ.front().first;
                int level = nodeQ.front().second;
                nodeQ.pop();
                
                if (level != curLevel) {
                    curLevel = level;
                    printf("\n");
                }
                
                for (auto &key : cur->keys){
                    cout<<key<<" ";
                }
                cout<<", ";
                
                for (auto &c : cur->childern){
                    if (c != nullptr) {
                        nodeQ.push({c, level+1});
                    }
                }
            }
        }
    };
}

#endif /* B_tree_hpp */
