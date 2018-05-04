//
//  TopK.hpp
//  algorithm
//
//  Created by shiwei on 18/5/4.
//
//

#ifndef TopK_hpp
#define TopK_hpp

#include <stdio.h>
#include "heap.hpp"

class TopK {
public:
    TFDataStruct::heap<int> *maxHeap = nullptr;
    int size = 0;
    TopK(int k) {
        size = k;
        maxHeap = new TFDataStruct::heap<int>(false);
    }
    
    void add(int num) {
        maxHeap->append(num);
    }
    
    vector<int> topk() {
        vector<int> result;
        auto limit = min((int)maxHeap->getValidSize(), size);
        
        for (int i = 0; i<limit; i++) {
            result.push_back(maxHeap->popTop());
        }
        for (auto val : result){
            maxHeap->append(val);
        }
        return result;
    }
};

#endif /* TopK_hpp */
