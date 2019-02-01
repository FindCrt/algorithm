//
//  main.m
//  algorithm
//
//  Created by shiwei on 17/8/16.
//
//

#include <stdio.h>
#include <math.h>
#include <vector>
#include <map>
#include <string>
#include <stack>
#include <queue>
#include <list>
#include <set>
#include <unordered_map>
#include <iostream>
#include <mach/mach_time.h>
#include <unordered_set>
#include <fstream>
#include <stdlib.h>
#include "Trie.hpp"

#include "CommonStructs.hpp"
#include "TypicalProblems.hpp"
#include "TestFuncs.hpp"



int maxSubArrayLen(vector<int> &nums, int k) {
    int sum = 0;
    int idx = 0;
    unordered_map<int, pair<int, int>> sumIdx;
    sumIdx[0] = {-1,-1};
    for (auto &n : nums) {
        sum += n;
        auto i = sumIdx.find(sum);
        if (i == sumIdx.end()) {
            sumIdx[sum] = {idx,idx};
        }else{
            i->second.first = min(i->second.first, idx);
            i->second.second = max(i->second.second, idx);
        }
        idx++;
    }
    
    int maxRange = 0;
    for (auto &p:sumIdx){
        auto i = sumIdx.find(p.first+k);
        if (i != sumIdx.end()) {
            maxRange = max(maxRange, i->second.second-p.second.first);
        }
    }
    
    return maxRange;
}

bool canPermutePalindrome(string &s) {
    bool pair[256];
    memset(pair, 0, sizeof(pair));
    
    for (auto &c : s){
        pair[c] = !pair[c];
    }
    
    int singleCount = 0;
    for (int i = 0; i<256;i++){
        singleCount += pair[i];
    }
    return singleCount<2;
}

class DataStream {
    struct ListNode{
        int val;
        ListNode *next = nullptr;
    };
    ListNode head;
    ListNode *tail = new ListNode();
    unordered_map<int, ListNode*> uniqueNodes;
public:
    
    DataStream(){
        head.next = tail;
    }
    
    void add(int num) {
        auto i = uniqueNodes.find(num);
        if (i == uniqueNodes.end()) {
            auto newNode = new ListNode();
            tail->val = num;
            tail->next = newNode;
            uniqueNodes[num] = tail;
            tail=newNode;
        }else{
            if (i->second->next == tail) {
                tail = i->second;
            }else{
                auto next = i->second->next;
                i->second->val = next->val;
                i->second->next = next->next;
                
                delete next;
                uniqueNodes[i->second->val] = i->second;
            }
            
            uniqueNodes.erase(i);
        }
    }

    int firstUnique() {
//        cout<<head.next<<" "<<head.next->val<<endl;
        return head.next->val;
    }
};

class RandomizedCollection {
    unordered_map<int, unordered_set<int>> idxes;
    vector<int> nums;
    int size = 0;
public:
    
    RandomizedCollection() {
        srandom((int)time(0));
    }
    
    bool insert(int val) {
        idxes[val].insert(size);
        nums.push_back(val);
        
        size++;
        return idxes[val].size()>1;
    }
    
    bool remove(int val) {
        auto &idxSet = idxes[val];
        if (idxSet.empty()) {
            return false;
        }
        
        auto removeIdx = *idxSet.begin();
        idxSet.erase(idxSet.begin());
        
        nums[removeIdx] = nums.back();
        nums.pop_back();
        
        size--;
        if (removeIdx != size) {
            auto last = idxes.find(nums[removeIdx]);
            last->second.erase(size);
            last->second.insert(removeIdx);
        }
        
        return true;
    }
    
    int getRandom() {
        auto idx = random()%nums.size();
        return nums[idx];
    }
};

int findMaxLength(vector<int> &nums) {
    unordered_map<int, int> diffIdx;
    
    int curDiff = 0;
    int idx = 0;
    diffIdx[0] = -1;
    
    int maxLen = 0;
    for (auto &n : nums){
        curDiff += n==0?-1:1;
        auto iter = diffIdx.find(curDiff);
        if (iter == diffIdx.end()) {
            diffIdx[curDiff]=idx;  //值保存第一个，这就是最前面的索引
        }else{
            //后面查找到具有相同差值(1的个数减0的个数)的，它们之间的区间可以构成解，所以更新最大长度
            maxLen = max(maxLen, idx - iter->second);
        }
        idx++;
    }
    return maxLen;
}

vector<string> subdomainVisits(vector<string> &cpdomains) {
    unordered_map<string, int> counts;
    for (auto &dom : cpdomains){
        int ct=0;
        int idx = 0, len = (int)dom.length();
        for (auto &c:dom){
            if (c==' ') {
                ct = rangeStringToInt(dom, 0, idx-1);
                counts[dom.substr(idx+1,len-idx-1)] += ct;
            }else if (c=='.'){
                counts[dom.substr(idx+1,len-idx-1)] += ct;
            }
            
            idx++;
        }
    }
    
    vector<string> result;
    for (auto &p:counts){
        result.push_back(to_string(p.second)+" "+p.first);
    }
    return result;
}

int numRabbits(vector<int> &answers) {
    unordered_map<int, int> groups;
    for (auto &ans:answers){
        groups[ans]++;
    }
    
    int num = (int)answers.size();
    for (auto &p:groups){
        int curNum = p.first+1;
        num += curNum-(p.second-1)%curNum-1;
    }
    return num;
}

int numJewelsInStones(string &J, string &S) {
    bool jewelry[52];
    memset(jewelry, 0, sizeof(jewelry));
    for (auto &c:J){
        int idx = c>'a'?c-'a':c-'A';
        jewelry[idx] = true;
    }
    
    int count = 0;
    for (auto &c:S){
        int idx = c>'a'?c-'a':c-'A';
        count += jewelry[idx];
    }
    
    return count;
}

vector<int> findErrorNums(vector<int> &nums) {
    int idx = 1;
    for (auto &n:nums){
        if (n!=idx) {
            return {n,idx};
        }
        idx++;
    }
    return {};
}

vector<int> smallestRange(vector<vector<int>> &nums) {
    vector<pair<int, int>> allNumbers;
    int idx = 0;
    //TODO: 可以用败者树做归并，少去排序的时间
    for (auto &v:nums){
        for (auto &n:v){
            allNumbers.push_back({n, idx});
        }
        idx++;
    }
    
    sort(allNumbers.begin(), allNumbers.end(), PairSort<int, int>::pairFirstComp);
    
    printVectorPair(allNumbers);
    
    int size = (int)allNumbers.size();
    int maxKind = (int)nums.size();
    int counts[maxKind];
    memset(counts, 0, sizeof(counts));
    int kindCount = 0;
    
    int i = 0, j = 0;
    int minStart=0, minRange = INT_MAX;
    
    while (1) {
        //开头前进，满足条件
        while (j<size && kindCount<maxKind) {
            do {
                if (++counts[allNumbers[j].second]==1) {
                    kindCount++;
                }
                j++;
            } while (j<size && allNumbers[j-1].first==allNumbers[j].first);
        }
        
        if (kindCount<maxKind) {
            break;
        }
        
        //尾部跟进，破坏条件，最后得到的i是第一个破坏条件的，也代表选i-1的值时是满足条件的最小长度
        while (kindCount>=maxKind) {
            do {
                if (--counts[allNumbers[i].second]==0) {
                    kindCount--;
                }
                i++;
            } while (allNumbers[i-1].first==allNumbers[i].first);
        }
        
        printf("(%d,%d):(%d,%d),%d\n",i-1,j-1,allNumbers[i-1].first,allNumbers[j-1].first, kindCount);
        //更新解
        int range = allNumbers[j-1].first-allNumbers[i-1].first;
        if (range<minRange) {
            minRange = range;
            minStart = allNumbers[i-1].first;
        }
    }
    
    return {minStart, minRange+minStart};
}

vector<vector<string>> findDuplicate(vector<string> &paths) {
    //为了在输出结果的时候，不需要从files里把vector拷贝到result,在集合里路径有两个之后，就直接把数据写入到result. files的value的状态变化如下：
    //初始：没有值；找到一个文件，这个文件的路径；找到两个文件及以上，result里存储文件路径集合的索引
    //最后的状态里，通过索引可以把路径直接存入到result里
    unordered_map<string, string> files;
    vector<vector<string>> result;
    
    for (auto &str:paths){
        int start = 0, idx = 0;
        int bracket1=0,bracket2=0;
        string dir;
        for (auto &c:str){
            if (c==' ') {
                if (start==0) {
                    dir = str.substr(start, idx-start)+"/";
                }else{
                    auto &p = files[str.substr(bracket1+1, bracket2-bracket1-1)];
                    if (p.empty()) {
                        p = dir+str.substr(start, bracket1-start);
                    }else if(p[0]=='r'){
                        result.push_back({p, dir+str.substr(start, bracket1-start)});
                        p = to_string(result.size()-1);
                    }else{
                        result[extractNumber(p)].push_back(dir+str.substr(start, bracket1-start));
                    }
                }
                
                start = idx+1;
            }else if (c=='('){
                bracket1 = idx;
            }else if (c==')'){
                bracket2 = idx;
            }
            idx++;
        }
        
        auto &p = files[str.substr(bracket1+1, bracket2-bracket1-1)];
        if (p.empty()) {
            p = dir+str.substr(start, bracket1-start);
        }else if(p[0]=='r'){
            result.push_back({p, dir+str.substr(start, bracket1-start)});
            p = to_string(result.size()-1);
        }else{
            result[extractNumber(p)].push_back(dir+str.substr(start, bracket1-start));
        }
    }
    
    return result;
}

int findPairs(vector<int> &nums, int k) {
    unordered_map<int, int> exist;
    for (auto &n:nums){
        exist[n]++;
    }
    
    int count = 0;
    if (k == 0) {
        for (auto &p:exist){
            if (p.second>1) {
                count++;
            }
        }
    }else{
        for (auto &p:exist){
            if (exist.find(p.first+k)!=exist.end()) {
                count++;
            }
        }
    }
    return count;
}

int islandPerimeter(vector<vector<int>> &grid) {
    int height = (int)grid.size(), width = (int)grid.front().size();
    int perimeter = 0;
    int i=0;
    for (auto &row:grid){
        int j=0;
        for (auto &mark:row){
            
            if (mark==1) {
                perimeter += (i==0||grid[i-1][j]==0)?1:0;
                perimeter += ((i==height-1)||grid[i+1][j]==0)?1:0;
                perimeter += (j==0||row[j-1]==0)?1:0;
                perimeter += ((j==width-1)||row[j+1]==0)?1:0;
            }
//            printf("(%d,%d) %d\n",i,j,perimeter);
            j++;
        }
        i++;
    }
    return perimeter;
}

int numberOfBoomerangs(vector<vector<int>> &points) {
    //格式：{点1的索引:{距离:点2的个数}}
    vector<unordered_map<int, int>> disMap;
    
    int i = 0;
    for (auto &p:points){
        disMap.push_back(unordered_map<int, int>());
        for (int j = i+1; j<points.size();j++){
            auto &q = points[j];
            int dis = (p[0]-q[0])*(p[0]-q[0])+(p[1]-q[1])*(p[1]-q[1]);
            disMap[i][dis]++;
            disMap[j][dis]++;
        }
        i++;
    }
    
    int count = 0;
    for (auto &p1:disMap){
        for (auto &p2:p1){
            count += p2.second*(p2.second-1);
        }
    }
    
    return count;
}

bool containsNearbyDuplicate(vector<int> &nums, int k) {
    unordered_map<int, int> idxMap;
    int idx = 0;
    for (auto &n:nums){
        auto iter = idxMap.find(n);
        if (iter != idxMap.end() && idx<iter->second) {
            return true;
        }
        idxMap[n] = idx+k+1;
        idx++;
    }
    
    return false;
}

int recommendFriends(vector<vector<int>> &friends, int user) {
    int scores[friends.size()];
    memset(scores, 0, sizeof(scores));
    
    for (auto &f:friends[user]){
        for (auto &f2:friends[f]){
            scores[f2]++;
        }
        scores[f]=0;
    }
    
    int maxScore=0, maxIdx=-1;
    for (int i = 0; i<friends.size(); i++) {
        if (i!=user && scores[i]>maxScore) {
            maxIdx = i;
            maxScore = scores[i];
        }
    }
    
    return maxIdx;
}

bool containsDuplicate(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    int last = INT_MIN;
    for(auto &n:nums){
        if (n==last) {
            return true;
        }
        last = n;
    }
    return false;
}

struct findingNumberNode {
    int val;
    int next;
};

int findingNumber(vector<vector<int>> &mat){
    
    unordered_map<int, bool> exist;
    for (auto &n:mat.front()){
        exist[n]=true;
    }
    
    int initSize = (int)exist.size();
    findingNumberNode candidates[initSize+1];
    
    int idx = 0;
    for (auto &p:exist){
        candidates[idx] = {p.first, idx+1};
        idx++;
    }
    candidates[initSize].next = -1; //最后是一个哨兵节点，用next==-1做标识
    
    for (auto mi = mat.begin()+1; mi!=mat.end();mi++){
        
        unordered_map<int, bool> exist;
        for (auto &n:*mi){
            exist[n]=true;
        }
        
        auto *cur = &candidates[0];
        while (cur->next>0) {  //next==-1时踩到了哨兵节点，结束遍历
            if (!exist[cur->val]) {
                //通过拷贝下一个节点的值实现单链表的删除
                cur->val = candidates[cur->next].val;
                cur->next = candidates[cur->next].next;
            }else{
                cur = &candidates[cur->next];
            }
        }
        if (candidates[0].next<0) {
            return -1;
        }
    }
    
    int result = INT_MAX;
    auto *cur = &candidates[0];
    while (cur->next>0) {  //next==-1时踩到了哨兵节点，结束遍历
        result = min(result, cur->val);
        cur = &candidates[cur->next];
    }
    return result;
}

vector<int> tree(vector<int> &x, vector<int> &y, vector<int> &a, vector<int> &b) {
    unordered_map<int, vector<int>> relations;
    for (int i = 0; i<x.size(); i++) {
        relations[x[i]].push_back(y[i]);
        relations[y[i]].push_back(x[i]);
    }
    
    unordered_map<int, int> parents;
    parents[1] = -1;
    
    stack<int> path;
    path.push(1);
    
    while (!path.empty()) {
        auto cur = path.top();
        path.pop();
        
        int parent = parents[cur];
        for (auto &e:relations[cur]){
            if (e != parent) {
                parents[e] = cur;
                path.push(e);
            }
        }
    }
    
    vector<int> result;
    for (int i = 0; i<a.size(); i++) {
        int val = 0;
        auto parA = parents[a[i]];
        auto parB = parents[b[i]];
        
        if (parA==b[i] || parB==a[i]) {
            val = 2;
        }else if(parA == parB){
            val = 1;
        }
        
        result.push_back(val);
    }
    
    return result;
}

vector<int> findSubstring(string &s, vector<string> &words) {
    
    typedef TFDataStruct::Trie<pair<int, int>> TrieType;
    TrieType exist;
    int len = (int)words.front().length();
    int wSize = (int)words.size();
    int totalLen = len*wSize;
    
    for (auto &w:words){
        exist.insert(w)->relateData.first++;
    }
    int sLen = (int)s.length();
    
    vector<int> result;
    for (int i = 0; i<=sLen-totalLen; i++) {
        
        exist.iterateNodes([](TrieType::TrieNode *node){
            node->relateData.second = node->relateData.first;
        });
        
        bool allExist = true;
        int score = 0;
        for (int j = 0; j<totalLen; j+=len) {
            auto node = exist.find(s.begin()+i+j, s.begin()+i+j+len);
            if (node == nullptr || node->count == 0) {
                allExist = false;
                break;
            }
            if (--node->relateData.second >= 0) {
                score++;
            }
        }
        if (allExist && score==wSize) {
            result.push_back(i);
        }
    }
    
    return result;
}

//一个式子由多个小式子构成，每个式子之间用运算符连接
//如果式子内只包含乘除，则弱化为组件，每个组件形式为:-1*a*b/c/d,即因子+乘变量+除变量
//class BCIComponent{
//public:
//    int factor = 0; //因子
//    vector<string> mulVarbs; //乘积变量
//    vector<string> divVarbs; //除的变量
//    BCIComponent(int factor = 0):factor(factor){};
//};
//
//class BCIFormula{
//public:
//    //复杂式子：subForms有多个子式子；简单式子：subForms为空，实际值为comp
//    vector<BCIFormula> subForms;
//    BCIComponent *comp = nullptr;
//    BCIFormula(int factor){
//        comp = new BCIComponent(factor);
//    }
//    BCIFormula(){};
//};

typedef enum{
    BCIFormulaOperatorAdd,
    BCIFormulaOperatorDiv,
    BCIFormulaOperatorMul,
}BCIFormulaOperator;

typedef enum{
    BCIElementTypeUnknown,
    BCIElementTypeAdd,
    BCIElementTypeSub,
    BCIElementTypeMul,
    BCIElementTypeDiv,
    BCIElementTypeBracketL,
    BCIElementTypeBracketR,
    BCIElementTypeVerb,
    BCIElementTypeNumber
}BCIElementType;

/*
 verb和subForm不共存，有它们之一时，值为factor*verb或factor*subForm，此时factor默认值为1
 这两个都不存在时，值就是factor,此时factor默认值为0
 */
class BCIFormula{
    struct MulDivRange{
        int start;
        int end;
    };
    vector<MulDivRange> MDRanges;
public:
    BCIFormulaOperator ope = BCIFormulaOperatorAdd;
    int factor = 1;
    string verb;
    vector<BCIFormula> subForms;
    
    BCIFormula(int factor, BCIElementType eType = BCIElementTypeAdd):factor(factor){
        setOpeWithEType(eType);
    };
    BCIFormula(string verb, BCIElementType eType = BCIElementTypeAdd):verb(verb){
        setOpeWithEType(eType);
    };
    BCIFormula(){};
    void setOpeWithEType(BCIElementType eType){
        switch (eType) {
            case BCIElementTypeMul:
                ope = BCIFormulaOperatorMul;
                break;
            case BCIElementTypeDiv:
                ope = BCIFormulaOperatorDiv;
                break;
            case BCIElementTypeSub:
                factor *= -1;
            default:
                ope = BCIFormulaOperatorAdd;
                break;
        }
    }
    
    //转化为字符串形式
    string show(){
        if (subForms.empty()) {
            if (verb.empty()) {
                return to_string(factor);
            }else{
                return factor == 1?verb:(to_string(factor)+"*"+verb);
            }
        }else{
            string exp="(";
            for (auto &form:subForms){
                if (form.ope == BCIFormulaOperatorAdd) {
                    if (form.factor>0) exp.push_back('+');
                }else if (form.ope == BCIElementTypeMul){
                    exp.push_back('*');
                }else{
                    exp.push_back('/');
                }
                
                exp += form.show();
            }
            exp.push_back(')');
            return exp;
        }
    }
    
    vector<string> showAddComps(){
        return {};
    }
    
    //简化式子。1. 常量融合 2.乘法多项式融合
    void simplify(){
        //先做乘法的融合，因为多出的式子会还需要加法融合，所以避免加法做两次
        int i=-1,j=0;
        int size = (int)subForms.size();
        while (j<size) {
            auto &ope = subForms[j].ope;
            if (i<0 && ope>BCIFormulaOperatorAdd) { //乘除
                i = j-1; //i作为乘除组的第一个式子
            }
            if (i>=0) { //有了第一个式子，找到后面的每个乘法因子融合
                if (ope==BCIFormulaOperatorMul) {
                    subForms[i]=multiplyMerge(subForms[i], subForms[j]);
                    subForms.erase(subForms.begin()+j);
                    j--;
                }else if(ope==BCIFormulaOperatorAdd){
                    i=-1;
                }
            }
            j++;
        }
    }
    
    //因为融合是内层在前面的，所以此时两个式子内部的子式子都是无法在拆解的
    static BCIFormula multiplyMerge(BCIFormula &form1, BCIFormula &form2){
        BCIFormula result;
        
        int i=-1,j=0;
        int size = (int)form1.subForms.size();
        while (j<size) {
            auto &ope = form1.subForms[j].ope;
            if (i<0 && ope>BCIFormulaOperatorAdd) {
                i = j-1;
            }else if(i>=0 && ope==BCIFormulaOperatorAdd){
                
                //[i,j-1]这是一个乘除的组合
                
                
                i=-1;
            }
            j++;
        }
        
        return result;
    }
};


//从start开始，一直解析到字符串结束或者遇到右括号，返回式子对象，结束为止填入到end
BCIFormula getFormulaFromExp(string &expression, unordered_map<string, int> &verbs, int start, int *end){
    
    BCIFormula formula;
//    formula.subForms.push_back(BCIFormula(0));
//    BCIFormula &pureNumForm = formula.subForms[0]; //零散的纯数
    
    int last = 0, len = (int)expression.length();
    BCIElementType lastEType = BCIElementTypeAdd;
    
    //把式子从字符串提取成对象
    for (int idx=start; idx<len; idx++){
        char c = expression[idx];
        if (c==' ') {
            BCIElementType eType = BCIElementTypeUnknown;
            if (idx-last==1) {
                char c = expression[last];
                switch (c) {
                    case '+':
                        eType = BCIElementTypeAdd;
                        break;
                    case '-':
                        eType = BCIElementTypeSub;
                        break;
                    case '*':
                        eType = BCIElementTypeMul;
                        break;
                    case '/':
                        eType = BCIElementTypeDiv;
                        break;
                    case '(':
                        eType = BCIElementTypeBracketL;
                        break;
                    case ')':
                        eType = BCIElementTypeBracketR;
                        break;
                    default:
                        break;
                }
            }
            
            if (eType == BCIElementTypeBracketL) {
                int end;
                BCIFormula nextForm = getFormulaFromExp(expression, verbs, idx+1, &end);
                nextForm.setOpeWithEType(lastEType);
                formula.subForms.push_back(nextForm);
                idx = end;
            }else if (eType == BCIElementTypeBracketR){ //这一段结束
                *end = idx;
                break;
            }else if (eType == BCIElementTypeUnknown) {
                bool isInt = false;
                int number = rangeStringToInt(expression, last, idx-1, &isInt);
                if (!isInt) {
                    auto verbName = expression.substr(last, idx-last);
                    auto valIter = verbs.find(verbName);
                    if (valIter != verbs.end()) {
                        formula.subForms.push_back(BCIFormula(valIter->second, lastEType));
                    }else{
                        formula.subForms.push_back(BCIFormula(verbName, lastEType));
                    }
                }else{
                    formula.subForms.push_back(BCIFormula(number, lastEType));
                }
            }
            
            lastEType = eType;
            last = idx+1;
        }
    }
    
    formula.simplify();
    
    return formula;
}

vector<string> basicCalculatorIV(string &expression, vector<string> &evalvars, vector<int> &evalints) {
    
    unordered_map<string, int> verbs;
    for (int i = 0; i<evalvars.size(); i++){
        verbs[evalvars[i]] = evalints[i];
    }
    
    expression.push_back(' ');
    
    int end;
    auto formula = getFormulaFromExp(expression, verbs, 0, &end);
    cout<<formula.show()<<endl;
    return formula.showAddComps();
}

int main(int argc, const char * argv[]) {
    uint64_t start = mach_absolute_time();
    string path = "/Users/apple/Downloads/9 (2).in";
    
    string exp = "e + 8 - a + 5";
    vector<string> evalvars = {"e"};
    vector<int> evalints = {1};
    basicCalculatorIV(exp, evalvars, evalints);
    
    
    uint64_t duration = mach_absolute_time() - start;
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double time = 1e-6 * (double)timebase.numer/timebase.denom * duration;
    printf("exe time: %.1f ms\n",time);
}
