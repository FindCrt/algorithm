//
//  CommonStructs.cpp
//  algorithm
//
//  Created by shiwei on 2018/9/1.
//

#include "CommonStructs.hpp"
#include <fstream>
#include <assert.h>

void readFile(string &path, const function<void(string &)>& handleLine){
    assert(handleLine);
    
    ifstream infile(path.c_str());
    if (!infile.is_open()){
        return;
    }
    
    string s;
    while (getline(infile, s)) {
        handleLine(s);
    }
    infile.close();
}
