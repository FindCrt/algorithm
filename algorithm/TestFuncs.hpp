//
//  TestFuncs.hpp
//  algorithm
//
//  Created by shiwei on 2019/1/17.
//

#ifndef TestFuncs_hpp
#define TestFuncs_hpp
#include "CommonStructs.hpp"

#include <stdio.h>
#include <vector>

inline int rangeStringToInt(string &str, int start, int end){
    int num = 0, digit = 1, sign = 1;
    if (str[start]=='-') {
        sign = -1;
        start++;
    }
    
    for (int i = end; i>=start; i--) {
        num += (str[i]-'0')*digit;
        digit *= 10;
    }
    
    return num*sign;
}

void readPoints(string &path, vector<Point> &points){
    readFile(path, [&points](string &line){
        
        int start=-1, idx = 0;
        int x = -1;
        for (auto &c : line){
            if (c == '[') {
                start = idx+1;
            }else if (c == ',' && start>0){
                x = rangeStringToInt(line, start, idx-1);
                start = idx+1;
            }else if (c == ']' && start>0){
                points.push_back({x, rangeStringToInt(line, start, idx-1)});
                start=-1;
            }
            
            idx++;
        }
    });
}

void readVectorInt(string &path, vector<int> &nums){
    readFile(path, [&nums](string &line){
        
        int start=-1, idx = 0;
        for (auto &c : line){
            if (c == '[') {
                start = idx+1;
            }else if (c == ',' && start>0){
                nums.push_back(rangeStringToInt(line, start, idx-1));
                start = idx+1;
            }else if (c == ']' && start>0){
                nums.push_back(rangeStringToInt(line, start, idx-1));
            }
            
            idx++;
        }
    });
}

void readVectorString(string &path, vector<string> &words){
    readFile(path, [&words](string &line){
        
        int start=-1, idx = 0;
        for (auto &c : line){
            if (c == '[') {
                start = idx+2;
            }else if (c == ',' && start>0){
                words.push_back(line.substr(start, idx-start-1));
                start = idx+2;
            }else if (c == ']' && start>0){
                words.push_back(line.substr(start, idx-start-1));
            }
            
            idx++;
        }
    });
}


void read2DVectorInt(string &path, vector<vector<int>> &nums){
    readFile(path, [&nums](string &line){
        
        int start=-1, idx = 0;
        vector<int> arr;
        for (auto &c : line){
            if (c == '[') {
                arr = {};
                start = idx+1;
            }else if (c == ',' && start>0){
                arr.push_back(rangeStringToInt(line, start, idx-1));
                start = idx+1;
            }else if (c == ']' && start>0){
                arr.push_back(rangeStringToInt(line, start, idx-1));
                nums.push_back(arr);
                start=-1;
            }
            
            idx++;
        }
    });
}

#endif /* TestFuncs_hpp */
