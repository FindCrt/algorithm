//
//  heap.hpp
//  algorithm
//
//  Created by shiwei on 17/11/28.
//
//

#ifndef heap_hpp
#define heap_hpp

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <math.h>

using namespace std;

#define leftChild(i) ((i<<1)+1)
#define rightChild(i) ((i<<1)+2)
#define parentIndex(i) ((i-1)>>1)

/** 逻辑是按照最小堆的来写的，这时比较函数是按按照a>b返回1来的。这样只要比较函数的逻辑翻转，就可以直接变成最大堆，只需要各处的处理逻辑保持一致 */
static size_t maxInitAllocSize = 1024;
namespace TFDataStruct {
    
    static int TFHeapComparePreUp = -1;     //让比较中的两个元素的前一个上浮
    static int TFHeapComparePreDown = 1;    //让比较中的两个元素的前一个下沉
    static int TFHeapCompareEqual = 0;      //两者相等
    
    template<class T>
    class heap {
        typedef int (*CompareFunc)(T a, T b);
        size_t _validSize = 0;      //实际有值得节点数
        size_t _limitSize = ULONG_MAX;      //限制节点数，超过这个数量，再添加就需要替换掉一个
        size_t _mallocSize = 8;     //申请的节点数
        
        
        T *_datas;
        
        CompareFunc _compare = NULL;
        
        inline void swap(size_t i, size_t j){
            T temp = _datas[i];
            _datas[i] = _datas[j];
            _datas[j] = temp;
        }
        
        static inline int defaultMaxHeapCompare(T a, T b){
            if (a < b) {
                return 1;
            }else if (a > b){
                return -1;
            }else{
                return 0;
            }
        }
        
        static inline int defaultMinHeapCompare(T a, T b){
            if (a < b) {
                return -1;
            }else if (a > b){
                return 1;
            }else{
                return 0;
            }
        }
        
        void sink_compare(size_t start){
            auto cur = start;
            while (rightChild(cur) < _validSize) {
                auto left = leftChild(cur), right = left+1;
                size_t min;
                if (_compare(_datas[left], _datas[right]) < 0) {
                    min = left;
                }else{
                    min = right;
                }
                
                if (_compare(_datas[cur], _datas[min]) < 0) {
                    break;
                }else{
                    swap(cur, min);
                    cur = min;
                }
            }
            
            auto left = leftChild(cur);
            if (left < _validSize && _compare(_datas[left], _datas[cur]) < 0) {
                swap(cur, left);
            }
        }
        
        void floatUp_compare(size_t start){
            auto cur = start;
            
            while (cur > 0) {
                auto parent = parentIndex(cur);
                if (_compare(_datas[parent], _datas[cur]) < 0) {
                    break;
                }else{
                    swap(cur, parent);
                    cur = parent;
                }
            }
        }
        
        inline void extend(){
            _mallocSize = 1 << ((int)log2(_mallocSize)+1);
            auto new_datas = (T*)malloc(_mallocSize*sizeof(T));
            memcpy(new_datas, _datas, _validSize* sizeof(T));
            _datas = new_datas;
        }
        
    public:
        
        
        heap(CompareFunc compare, size_t limitSize = ULONG_MAX, vector<T> *vec = nullptr){
            
            _limitSize = limitSize;
            if (limitSize > 0) {
                _mallocSize = min(maxInitAllocSize, limitSize);
            }
            
            _datas = (T*)malloc(_mallocSize*sizeof(T));
            
            _compare = compare;
            
            if (vec != nullptr) {
                for (auto iter = vec->begin(); iter != vec->end(); iter++) {
                    append(*iter);
                }
            }
        };
        
        heap(bool isMinHeap, size_t limitSize = ULONG_MAX, vector<T> *vec = nullptr){
            new (this) heap(isMinHeap ? defaultMinHeapCompare : defaultMaxHeapCompare, limitSize, vec);
        }
        
        size_t getValidSize(){
            return _validSize;
        }
        
        size_t getLimitSize(){
            return _limitSize;
        }
        
        bool isEmpty(){
            return _validSize == 0;
        }
        
        T getTop(){
            return _datas[0];
        }
        
        T getDataAtIndex(size_t index){
            return _datas[index];
        }
        
        //size会增加
        void append(T node){
            _validSize++;
            if (_validSize > _mallocSize) {
                extend();
            }
            
            _datas[_validSize-1] = node;
            floatUp_compare(_validSize-1);
            if (_validSize > _limitSize) {
                _validSize--;
            }
        };
        
        void remove(size_t index){
            _datas[index] = _datas[_validSize-1];
            _validSize--;
            update(_datas[index], index);
        }
        
        //size不变，一个新的进来，挤掉一个旧的
        void replace(T node, size_t index){
            //TODO: index可能越界
            _datas[index] = node;
            update(node, index);
        }
        
        //某个节点的数据改变，更新它的位置
        void update(T node, long index = -1){
            if (index < 0) {
                for (int i = 0; i<_validSize; i++) {
                    if (_datas[i] == node) {
                        index = i;
                    }
                }
                
                if (index < 0) {
                    printf("未找到要更新节点\n");
                    return;
                }
            }
            if (index > 0 && _compare(node, _datas[parentIndex(index)]) < 0) { //按最小堆逻辑，小于，则上浮
                
                floatUp_compare(index);
                
            }else if ((leftChild(index) < _validSize && _compare(node,_datas[leftChild(index)]) > 0) ||
                      (rightChild(index) < _validSize && _compare(node,_datas[rightChild(index)])) > 0){
                
                //大于子节点中任何一个，就不稳定，需要下浮
                sink_compare(index);
            }
        }
        
        T popTop(){
            
            if (_validSize == 0) abort();
            T top = _datas[0];
            _datas[0] = _datas[_validSize-1];
            _validSize--;
            sink_compare(0);
            
            return top;
        }
        
        friend ostream& operator<<(ostream& os, heap &heap){
            for (int i = 0; i<heap._validSize; i++) {
                os<<heap._datas[i]<<" ";
            }
            
            return os;
        }
        
    };
}

#endif /* heap_hpp */
