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
            insert(midKey, node->vals[leftCount], node->parent, pKeyIdx);
            
            //新建一个节点作为分裂后的右节点
            NodeType *other = new NodeType();
            other->size = rightCount+1;
            other->parent = node->parent;
            other->keys.insert(other->keys.begin(), node->keys.begin()+leftCount+1, node->keys.end());
            other->vals.insert(other->vals.begin(), node->vals.begin()+leftCount+1, node->vals.end());
            other->childern.insert(other->childern.begin(), node->childern.begin()+leftCount+1, node->childern.end());
            node->parent->childern[pKeyIdx+1] = other;
            
            //原节点内容缩小，变成分裂后的左节点
            node->size = leftCount+1;
            node->keys.erase(node->keys.begin()+leftCount, node->keys.end());
            node->vals.erase(node->vals.begin()+leftCount, node->vals.end());
            node->childern.erase(node->childern.begin()+leftCount+1, node->childern.end());
        }
        
        //在指定节点的指定位置插入关键字
        void insert(keyTp &key, valTp &val, NodeType *node, int keyIdx){
            
            node->size++;
            node->keys.insert(node->keys.begin()+keyIdx, key);
            node->vals.insert(node->vals.begin()+keyIdx, val);
            node->childern.insert(node->childern.begin()+keyIdx+1, nullptr);
            
            //超过最大子节点个数，分裂
            if (node->size > maxSize) {
                divide(node);
            }
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
                
                insert(key, val, node, keyIdx);
            }
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
