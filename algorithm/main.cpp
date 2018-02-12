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
#include <iostream>
#include "heap.hpp"
#include <mach/mach_time.h>

#include "TFSort.h"
#include "MinStack.hpp"

using namespace std;

class ListNode {
    public:
    int val;
    ListNode *next;
    ListNode(int val) {
        this->val = val;
        this->next = NULL;
    }
    
    static ListNode *createList(vector<int> vals){
        ListNode *head = nullptr;
        ListNode *last = nullptr;
        for (auto iter = vals.begin(); iter != vals.end(); iter ++) {
            ListNode *node = new ListNode(*iter);
            
            if (head == nullptr) {
                head = node;
            }else{
                last->next = node;
            }
            last = node;
        }
        
        return head;
    }
    
    static void printList(ListNode *list){
        if (list == nullptr) {
            printf("list is empty!\n");
            return;
        }
        
        printf("%d",list->val);
        list = list->next;
        
        while (list) {
            printf("->%d",list->val);
            list = list->next;
        }
        
        printf("\n");
    }
};

class TreeNode{
public:
    int val;
    TreeNode *left, *right;
    TreeNode(int val){
        this->val = val;
        this->left = this->right = NULL;
    }
};

int addDigits(int num) {
    
    while (num > 9) {
        int sum = 0;
        while (num > 9) {
            sum += num % 10;
            num /= 10;
        }
        sum += num;
        num = sum;
    }
    
    return num;
}

inline int select(int m, int n){
    if (n>m) {
        return 0;
    }
    if (n == 0) {
        return 1;
    }
    
    double sum1 = 1, sum2 = 1;
    for (int i = 0; i<m-n; i++) {
        sum1 *= n+i+1;
        sum2 *= i+1;
    }
    
    return sum1/sum2;
}

int numWays(int n, int k) {
    int sum = 0;
    for (int i = 0; i<n; i++) {
        sum += (k * pow(k-1, n-i-1)) * select(n-i, i);
    }
    
    return sum;
}

bool isHappy(int n) {
    map<int,bool> mark;
    while (n != 1) {
        if (mark[n]) {
            return false;
        }
        mark[n] = true;
        int sum = 0;
        while (n > 9) {
            int single = n % 10;
            n /= 10;
            sum += single * single;
        }
        sum += n * n;
        n = sum;
    }
    
    return true;
}

vector<string> longestWords(vector<string> &dictionary) {
    // write your code here
    vector<string> result;
    size_t maxLength = 0;
    for (auto iter = dictionary.begin(); iter != dictionary.end(); iter++) {
        if (iter->length() > maxLength) {
            maxLength = iter->length();
            result.clear();
            result.push_back(*iter);
        }else if (iter->length() == maxLength){
            result.push_back(*iter);
        }
    }
    
    return result;
}

int numIslands(vector<vector<bool>>& grid) {
    // Write your code here
    int islandNum = 0;
    int islandTag = 1024;
    map<int, int> islandEdge;
    for (size_t i = 0; i<grid.size(); i++) {
        auto row = grid[i];
        map<int, int>curEdge;
        
        vector<vector<int>*> rowIsland;
        vector<int> *oneIsland = new vector<int>();
        for (int j = 0; j<row.size(); j++) {
            int isIsland = row[j];
            
            if (isIsland) {
                oneIsland->push_back(j);
                //curEdge[j] = islandTag;
            }else if(!oneIsland->empty()){
                rowIsland.push_back(oneIsland);
                oneIsland = new vector<int>();
                //islandTag ++;
            }
        }
        if(!oneIsland->empty()){
            rowIsland.push_back(oneIsland);
            //islandTag ++;
        }
        
        for (auto iter = rowIsland.begin(); iter != rowIsland.end(); iter++) {
            auto oneIsland = *iter;
            
            vector<int> links;
            for (auto iter2 = oneIsland->begin(); iter2 != oneIsland->end(); iter2++) {
                int link = islandEdge[*iter2];
                if (link && find(links.begin(), links.end(), link) == links.end()) {
                    links.push_back(link);
                }
            }
            
            if (links.empty()) {
                islandNum ++;
                islandTag ++;
                
                for (auto iter2 = oneIsland->begin(); iter2 != oneIsland->end(); iter2++){
                    curEdge[*iter2] = islandTag;
                }
            }else if(links.size() == 1){
                for (auto iter2 = oneIsland->begin(); iter2 != oneIsland->end(); iter2++){
                    curEdge[*iter2] = links[0];
                }
            }else{
                islandNum -= links.size() - 1;
                for (auto iter2 = oneIsland->begin(); iter2 != oneIsland->end(); iter2++){
                    curEdge[*iter2] = links[0];
                }
            }
        }
        
        islandEdge = curEdge;
        islandTag ++;
    }
    
    return islandNum;
}

bool isValidParentheses(string s) {
    // write your code here
    stack<char> unpaired;
    for (auto iter = s.begin(); iter != s.end(); iter++) {
        if (*iter == '{' || *iter == '[' || *iter == '(') {
            unpaired.push(*iter);
        }else if (*iter == '}'){
            if (unpaired.top() != '{') {
                return false;
            }else{
                unpaired.pop();
            }
        }else if (*iter == ']'){
            if (unpaired.top() != '[') {
                return false;
            }else{
                unpaired.pop();
            }
        }else if (*iter == ')'){
            if (unpaired.top() != '(') {
                return false;
            }else{
                unpaired.pop();
            }
        }
    }
    
    return unpaired.empty();
}

int lengthOfLastWord(string s) {
    // write your code here
    int length = 0;
    bool inBlank = true;
    for (auto iter = s.begin(); iter != s.end(); iter++) {
        if (*iter != ' ') {
            length ++;
            if (inBlank) {
                length = 1;
                inBlank = false;
            }
            
        }else{
            inBlank = true;
        }
    }
    
    return length;
}

string countAndSay(int n) {
    // write your code here
    string nums = "1";
    for (int i = 1; i < n; i++) {
        char curChar = 0;
        int curCount = 0;
        string tempNums = "";
        for (auto iter = nums.begin(); iter != nums.end(); iter++) {
            if (*iter != curChar) {
                if (curChar) {
                    tempNums += to_string(curCount);
                    tempNums += curChar;
                }
                curChar = *iter;
                curCount = 1;
            }else{
                curCount ++;
            }
        }
        
        if (curCount) {
            tempNums += to_string(curCount);
            tempNums += curChar;
        }
        
        nums = tempNums;
        
        //printf("%d\n",i);
    }
    
    return nums;
}

inline bool isAlphabet(char c){
    return (c <= 'z' && c >= 'a') || (c <= 'Z' && c >= 'A');
}

inline bool isNumber(char c){
    return c <= '9' && c >= '0';
}

bool isPalindrome(string s) {
    
    bool isBlank = true;
    for (auto iter = s.begin(); iter != s.end(); iter++) {
        if (*iter != ' ') {
            isBlank = false;
            break;
        }
    }
    
    if (isBlank) {
        return true;
    }
    
    auto iterLeft = s.begin();
    auto iterRight = s.end()-1;
    while (iterLeft <= iterRight) {
        
        while (!isNumber(*iterLeft) && !isAlphabet(*iterLeft) && iterLeft != s.end()) {
            iterLeft ++;
        }
        if (iterLeft == s.end()) {
            break;
        }
        while (!isNumber(*iterRight) && !isAlphabet(*iterRight) ) {
            iterRight --;
        }
        
        if (tolower(*iterLeft) != tolower(*iterRight)) {
            return false;
        }else{
            iterLeft ++;
            iterRight --;
        }
    }
    
    return true;
}

inline void rotateStringRange(string &str, string::iterator first, string::iterator last){
    auto left = first;
    auto right = last-1;
    while (left < right) {
        char temp = *left;
        *left = *right;
        *right = temp;
        
        left ++;
        right --;
    }
}

void rotateString(string &str,int offset){
    if (str.empty()) {
        return;
    }
    
    offset %= str.length();
    
    auto border = str.begin()+ (str.length() - offset);
    rotateStringRange(str, str.begin(), border);
    rotateStringRange(str, border, str.end());
    rotateStringRange(str, str.begin(), str.end());
}

int aplusb(int a, int b) {
    int size = sizeof(int) * 8;
    int c = 0;
    int carry = 0;
    for (int i = 0; i<size; i++) {
        int bitA = a & (1 << i);
        int bitB = b & (1 << i);
        
        int result = bitA ^ bitB ^ carry;
        int andAB = bitA & bitB & carry;
        int orAB = bitA | bitB | carry;
        
        if ((result && andAB) || (!result && orAB)) {  //carry
            carry = 1 << (i+1);
        }else{
            carry = 0;
        }
        
        if (result) {
            c |= result;
        }
    }
    
    return c;
}

long long trailingZeros(long long n) {
    long long num = 0;
    while (n >= 5) {
        n /= 5;
        num += n;
    }
    return num;
}

vector<int> mergeSortedArray(vector<int> A, vector<int> B) {
    auto iterA = A.begin();
    auto iterB = B.begin();
    vector<int> merge;
    
    while (iterA != A.end() && iterB != B.end()) {
        if (*iterA < *iterB) {
            merge.push_back(*iterA);
            iterA ++;
        }else{
            merge.push_back(*iterB);
            iterB ++;
        }
    }
    
    while (iterA != A.end()) {
        merge.push_back(*iterA);
        iterA ++;
    }
    while (iterB != B.end()) {
        merge.push_back(*iterB);
        iterB ++;
    }
    
    return merge;
}

template<class T>
inline void rotateVectorRange(vector<T> &vec, typename vector<T>::iterator first, typename vector<T>::iterator last){
    auto left = first;
    auto right = last-1;
    while (left < right) {
        char temp = *left;
        *left = *right;
        *right = temp;
        
        left ++;
        right --;
    }
}

template<class T>
void rotateString(vector<T> &vec,int offset){
    if (vec.empty()) {
        return;
    }
    
    offset %= vec.size();
    
    auto border = vec.begin()+ (vec.length() - offset);
    rotateStringRange(vec, vec.begin(), border);
    rotateStringRange(vec, border, vec.end());
    rotateStringRange(vec, vec.begin(), vec.end());
}

void recoverRotatedSortedArray(vector<int> nums) {
    int left = 0, right = (int)nums.size()-1;
    while (left < right - 1) {
        int mid = (left+right)/2;
        if (mid < nums[right]) {
            right = mid;
        }else if (mid > nums[left]){
            left = mid;
        }
    }
}

int majorityNumber(vector<int> nums) {
    int num = 0;
    int count = 0;
    for (auto iter = nums.begin(); iter != nums.end(); iter++) {
        if (num == *iter) {
            count ++;
        }else if (count == 0){
            num = *iter;
            count = 1;
        }else{
            count --;
        }
    }
    
    return num;
}

vector<long long> productExcludeItself(vector<int> nums) {
    vector<long long>result;
    if (nums.empty()) {
        return result;
    }else if (nums.size() == 1){
        result.push_back(1);
        return result;
    }
    
    vector<long long>leftProducts;
    vector<long long>rightProducts;
    
    long long left = 1;
    for (auto iter = nums.begin(); iter != nums.end()-1; iter++) {
        left *= *iter;
        leftProducts.push_back(left);
    }
    
    long long right = 1;
    for (auto iter = nums.end() - 1; iter > nums.begin(); iter--) {
        right *= *iter;
        rightProducts.push_back(right);
    }
    
    
    result.push_back(rightProducts[nums.size()-2]);
    for (int i = 1; i< nums.size()-1; i++) {
        result.push_back(leftProducts[i-1] * rightProducts[nums.size() - i - 2]);
    }
    result.push_back(leftProducts[nums.size()-2]);
    
    return result;
}

inline int partionLeft(vector<int> &nums, int first, int last){
    int base = nums[first];
    int i = first;
    for (int j = first + 1; j <= last; j++) {
        if (nums[j] <= base) {
            i++;
            
            int temp = nums[j];
            nums[j] = nums[i];
            nums[i] = temp;
        }
    }

    nums[0] = nums[i];
    nums[i] = base;
    
    return i;
}

int median(vector<int> &nums) {
    
    int first = 0, last = (int)nums.size() -1, part = -1, mid = ((int)nums.size()-1)/2;
    do {
        part = partionLeft(nums, first, last);
        if (part < mid) {
            first = part + 1;
        }else if (part > mid){
            last = part - 1;
        }else{
            printf("mid: %d \n",mid);
            return nums[mid];
        }
    } while (1);
}

int singleNumber(vector<int> A) {
    if (A.empty()) {
        return 0;
    }
    int result = A.front();
    for (auto iter = A.begin()+1; iter != A.end(); iter++) {
        result ^= *iter;
    }
    
    return result;
}

string reverseWords(string s) {
    int left = 0, right = 0, blank = 0;
    int cur = 0;
    bool inBlank = true;
    while (cur < s.size()) {
        if (s[cur] != ' ' && inBlank) {  //发现非空
            blank = cur - left;            //会被翻转到下一阶段的空格，减去了留存的一个空格
            inBlank = false;
        }else if (s[cur] == ' ' && !inBlank){   //进入空格
            right = cur - 1;
            inBlank = true;
            
            //reverse one world
            auto iter1 = s.begin() + left, iter2 = s.begin()+right;
            while (iter1 < iter2) {
                int temp = *iter1;
                *iter1 = *iter2;
                *iter2 = temp;
                
                iter1 ++;
                iter2 --;
            }
            
            left = right - (blank-2);
        }
        
        cur ++;
    }
    
    if (!inBlank) {
        right = cur-1;
        auto iter1 = s.begin() + left, iter2 = s.begin()+right;
        while (iter1 < iter2) {
            int temp = *iter1;
            *iter1 = *iter2;
            *iter2 = temp;
            
            iter1 ++;
            iter2 --;
        }
        
        s.erase(s.end()-blank, s.end());
        
    }else{
        s.erase(s.begin()+left-1, s.end());
    }
    
    return s;
}

//1. 进制计算形式，累计计算 2.对33进行位运算 3. 最终结果是取余，可以不断取余来压缩数的范围
int hashCode(string key,int HASH_SIZE) {
    unsigned long hash = 0;
    for (auto iter = key.begin(); iter != key.end(); iter++) {
        hash = (hash << 5)%HASH_SIZE + hash + (unsigned long)*iter;
        printf("%ld\n",hash);
    }
    
    return hash % HASH_SIZE;
}

vector<int> subarraySum(vector<int> nums){
    map<int, int>sumIndexMap;
    
    int sum = 0;
    for (int i = 0; i<nums.size(); i++) {
        sum += nums[i];
        if (sumIndexMap.find(sum) == sumIndexMap.end()) {
            if (sum == 0) {
                return {0, i};
            }
            sumIndexMap[sum] = i;
        }else{
            return {sumIndexMap[sum]+1, i};
        }
    }
    
    return {};
}

ListNode *insertionSortList(ListNode *head) {
    if (head == nullptr) {
        return head;
    }
    
    ListNode *willInsert = head->next;  //将要被插入的节点
    ListNode *sortedTrail = head;       //已拍好序的节点结尾
    while (willInsert) {
        ListNode *posRight = head;      //找到的插入位置右边
        ListNode *posLeft = nullptr;    //找到的插入位置左边
        while (posRight != willInsert && posRight->val <= willInsert->val) {
            posLeft = posRight;
            posRight = posRight->next;
        }
        
        if (posRight != willInsert) {
            
            sortedTrail->next = willInsert->next;
            
            if (posLeft) {
                posLeft->next = willInsert; //中间
            }else{
                head = willInsert; //左边界
            }
            
            willInsert->next = posRight;
        }else{
            sortedTrail = sortedTrail->next;  //右边界更新已排序结尾
        }
        
//        printf("%d ",willInsert->val);
        willInsert = sortedTrail->next;
    }
    
    return head;
}

int removeElement(vector<int> &A, int elem) {
    
    int size = 0;
    
    auto slowIter = A.begin(), fastIter = A.begin();
    while (fastIter != A.end()) {
        
        if (*fastIter != elem) {
            *slowIter = *fastIter;
            slowIter ++;
            size ++;
        }
        
        fastIter ++;
    }
    
    return size;
}

vector<int> plusOne(vector<int> &digits) {
    
    vector<int> result;
    int carry = 1;
    for (int i = (int)digits.size()-1; i >= 0; i--) {
        int bit = digits[i]+carry;
        if (bit > 9) {
            carry = 1;
            bit -= 10;
        }else{
            carry = 0;
        }
        result.insert(result.begin(), bit);
    }
    
    if (carry) {
        result.insert(result.begin(), 1);
    }
    
    return result;
}

int reverseInteger(int n) {
    int negative = 1;
    if (n < 0) {
        negative = -1;
        n = -n;
    }
    
    vector<int> bits;
    while (n >= 1) {
        bits.push_back(n%10);
        n /= 10;
    }
    
    int result = 0;
    int power = 1;
    for (int i = (int)bits.size()-1; i >= 0 ; i--) {
        result += bits[i] * power;
        power *= 10;
    }
    
    return result * negative;
}


ListNode *removeNthFromEnd(ListNode *head, int n) {
    if (head == nullptr) {
        return nullptr;
    }
    
    int count = 0;
    ListNode *cur = head;
    while (cur) {
        count ++;
        cur = cur->next;
    }
    
    if (n == count) {
        return head->next;
    }
    
    int i = 0;
    cur = head;
    while (i != count - n -1) {
        i++;
        cur = cur->next;
    }
    
    cur->next = cur->next->next;
    
    return head;
}

int bitSwapRequired(int a, int b) {
    int XOR = a ^ b;
    
//    int size = sizeof(int)*8;
    int count = 0;
    while (XOR) {
        count += (XOR & 1);
        XOR = XOR >> 1;
//        size --;
    }
    
    return count;
 }

string concatenetedString(string &s1, string &s2) {
    vector<bool> unexistIndex;
    for (int i = 0; i<s2.length(); i++) {
        unexistIndex.push_back(true);
    }
    
    string result;
    
    for (auto iter = s1.begin(); iter != s1.end(); iter++) {
        bool exist = false;
        for (int j = 0; j < s2.length(); j++) {
            if (*iter == s2[j]) {
                exist = true;
                unexistIndex[j] = false;
            }
        }
        
        if (!exist) {
            result.push_back(*iter);
        }
    }
    
    for (int j = 0; j < unexistIndex.size(); j++) {
        if (unexistIndex[j]) {
            result.push_back(s2[j]);
        }
    }
    
    return result;
}

bool checkSumOfSquareNumbers(int num) {
    if (num == 0) {
        return true;
    }
    int rate = 1;
    while (num % 4 == 0) {
        num = num/4;
        rate *= 4;
    }
    printf("rate: %d\n",rate);
    
    double mid = sqrt(num/2);
    int max = sqrt(num);
    int left = floor(mid);
    int right = ceil(mid);
    
    while (left >= 0 && right <= max) {
        int square = left * left + right * right;
        if (square == num) {
            printf("num:%d left:%D right: %d\n",num*rate,left*rate/2,right*rate/2);
            return true;
        }else if (square < num){
            right ++;
        }else{
            left --;
        }
    }
    
    return false;
}

typedef struct{
    int start;
    int end;
}Range;

vector<string> missingString(string str1, string str2) {
    
    vector<string> missingStrs;
    
    bool missing = false;
    string missingStr;
    int i = 0, j= 0;
    
    while (i < str1.size() && j < str2.size()) {
        
        if (str1[i] != str2[j]) {
            if (!missing) {
                missing = true;
            }
            
            missingStr.push_back(str1[i]);
            
            i++;
            
        }else{
            if (missing) {
                missing = false;
                missingStrs.push_back(missingStr);
                missingStr = "";
            }
            
            i++;
            j++;
        }
    }
    
    if (!missing && j == str2.size() && i < str1.size()) {
        missingStrs.push_back(str1.substr(i,str1.size() - i));
    }
    
    if (missing) {
        printf("wrong str2");
        missingStrs.push_back(missingStr);
    }
    
    return missingStrs;
}

int splitString(string& s, int start, vector<vector<string>> &split){
    
    if (start == s.size()-1) {
        split.push_back({s.substr(start,1)});
        return 1;
    }else if (start == s.size()){
        return 0;
    }
    
    int count1 = splitString(s, start+1, split);
    for (auto iter = split.end() - count1; iter != split.end(); iter++) {
        iter->insert(iter->begin(),(s.substr(start,1)));
    }
    
    int count2 = splitString(s, start+2, split);
    if (count2 == 0) {
        split.push_back({s.substr(start, 2)});
        count2 = 1;
    }else{
        for (auto iter = split.end() - count2; iter != split.end(); iter++) {
            iter->insert(iter->begin(),s.substr(start,2));
        }
    }
    
    return count1 + count2;
}

vector<vector<string>> splitString(string& s) {
    if (s.length() == 0) {
        return {{}};
    }
    vector<vector<string>> result;
    
    splitString(s, 0, result);
    
    return result;
}

bool isIsomorphic(string s, string t) {
    char map[256];
    char reverseMap[256];
    
    memset(map, 0, sizeof(map));
    memset(reverseMap, 0, sizeof(reverseMap));
    
    for (int i = 0; i<s.size(); i++) {
        int index = s[i];
        
        char mapTo = map[index];
        if (mapTo) {
            if (mapTo != t[i]) {
                return false;
            }
        }else{
            if (reverseMap[t[i]]) {  //两个字符map到同一个字符
                return false;
            }
            map[index] = t[i];
            reverseMap[t[i]] = index;
        }
    }
    
    return true;
}

//在整个字符串里唯一的字符，找到符合这个条件的第一个
int firstUniqChar(string &s) {
    
    int charCount = 256;
    int exist[charCount];
    memset(exist, 0, sizeof(exist));
    
    for (auto iter = s.begin(); iter != s.end(); iter++) {
        int index = *iter;
        
        if (exist[index] == 0) {
            exist[index] = 1;
        }else if (exist[index] == 1){
            exist[index] = 2;
        }
    }
    
    for (int i = 0; i<s.length();i++) {
        
        int index = s[i];
        if (exist[index] == 1) {
            return i;
        }
    }
    
    return -1;
}


int guess(int num){
    int right = 2147483647;
    if (num < right) {
        return -1;
    }else if (num > right){
        return 1;
    }else{
        return 0;
    }
}

int guessNumber(int n) {
    int left = 1, right = n;
    while (left < right) {
        int mid = left + (right - left)/2;
//        printf("[%d,%d]->%d\n",left, right, mid);
        int compare = guess(mid);
        if (compare == -1) {
            left = mid+1;
        }else if (compare == 1){
            right = mid-1;
        }else{
            return mid;
        }
    }
    
    return left;
}


/**** 不使用额外空间，代码简洁 *****/
void calculateTreeSum(TreeNode *root){
    if (root == nullptr) {
        return ;
    }
    
    calculateTreeSum(root->left);
    calculateTreeSum(root->right);
    
    
    
    root->val = (root->left == nullptr ? 0:root->left->val )+ (root->right == nullptr ? 0:root->right->val)+ root->val;
}

void convertBST(TreeNode *root, TreeNode *firstRightParent){
    
    if (root == nullptr) {
        return;
    }
    
    root->val = (firstRightParent == nullptr ? 0:firstRightParent->val) + (root->val - (root->left == nullptr ? 0:root->left->val));
    
    convertBST(root->right, firstRightParent);
    convertBST(root->left, root);
}

TreeNode * convertBST(TreeNode * root) {
    //先全部改成子树总值
    calculateTreeSum(root);
    
    convertBST(root, nullptr);
    
    return root;
}

string leftPad(string &originalStr, int size, char padChar=' ') {
    char pad[size-originalStr.length()];
    memset(pad, padChar, sizeof(pad));
    string result = pad;
    result.append(originalStr);
    
    return result;
}

bool isIdentical(TreeNode * a, TreeNode * b) {
    if (a == nullptr && b == nullptr) {
        return true;
    }else if (a == nullptr || b == nullptr){
        return false;
    }
    
    if (a->val != b->val) {
        return false;
    }
    
    return isIdentical(a->left, b->left) && isIdentical(a->right, b->right);
}

int findPosition(vector<int> &nums, int target) {
    int left = 0, right = (int)nums.size()-1;
    
    while (left <= right) {
        int mid = left + (right-left)/2;
        if (nums[mid] < target) {
            left = mid + 1;
        }else if (nums[mid] > target){
            right = mid - 1;
        }else{
            return mid;
        }
    }
    
    return -1;
}

//int longestIncreasingContinuousSubsequence(vector<int> &A, bool increase) {
//    int decreaseIndex = 0;
//    int curIndex = 1;
//    auto lastIter = A.begin();
//    
//    int maxLen = 0;
//    for (auto iter = A.begin()+1; iter != A.end(); iter++) {
//        if ((*iter > *lastIter) == increase) {
//            maxLen = max(maxLen, curIndex - decreaseIndex);
//            decreaseIndex = curIndex;
//        }
//        
//        lastIter = iter;
//        curIndex++;
//    }
//    
//    //结尾算作一组
//    maxLen = max(maxLen, curIndex - decreaseIndex);
//    
//    return maxLen;
//}


//int longestIncreasingContinuousSubsequence(vector<int> &A) {
//    if (A.empty()) {
//        return 0;
//    }
//    int increaseMax = longestIncreasingContinuousSubsequence(A, true);
//    int decreaseMax = longestIncreasingContinuousSubsequence(A, false);
//    
//    return max(increaseMax, decreaseMax);
//}

int longestIncreasingContinuousSubsequence(vector<int> &A, int start, bool increase){
    if (start == A.size() - 1) {
        return 1;
    }
    int lower = longestIncreasingContinuousSubsequence(A, start+1, increase);
    
    int maxLen = 1;
    for (int i = start+1; i < A.size(); i++) {
        if ((A[i] > A[i-1]) == increase) {
            maxLen++;
        }else{
            break;
        }
    }
    
    return max(maxLen, lower);
}

int longestIncreasingContinuousSubsequence(vector<int> &A){
    if (A.empty()) {
        return 0;
    }
    int increaseMax = longestIncreasingContinuousSubsequence(A, 0, true);
    int decreaseMax = longestIncreasingContinuousSubsequence(A, 0, false);
    
    return max(increaseMax, decreaseMax);
}

void partitionArray(vector<int> &nums) {
    if (nums.empty()) {
        return;
    }
    auto leftIter = nums.begin();
    auto rightIter = nums.end()-1;
    
    bool leftMove = true;
    while (leftIter < rightIter) {
        if (leftMove) {
            if (!(*leftIter & 1)) { //偶数 x & 1 可以把最后一位取出来
                leftMove = false;
            }else{
                leftIter++;
            }
        }else{
            if (*rightIter & 1) {
                leftMove = true;
                
                auto temp = *leftIter;
                *leftIter = *rightIter;
                *rightIter = temp;
                
                leftIter++;
            }
            rightIter--;
        }
    }
}

//分割
void partitionArray(vector<int> &nums, bool(*shouMoveToLeft)(int num)){
    if (nums.empty()) {
        return;
    }
    auto leftIter = nums.begin();
    auto rightIter = nums.end()-1;
    
    bool leftMove = true;
    while (leftIter < rightIter) {
        if (leftMove) {
            if (!shouMoveToLeft(*leftIter)) {
                leftMove = false;
            }else{
                leftIter++;
            }
        }else{
            if (shouMoveToLeft(*rightIter)) {
                leftMove = true;
                
                auto temp = *leftIter;
                *leftIter = *rightIter;
                *rightIter = temp;
                
                leftIter++;
            }
            rightIter--;
        }
    }
}

bool isOdd(int num){
    return num & 1;
}

class Interval {
public:
    int start, end;
    Interval(int start, int end) {
        this->start = start;
        this->end = end;
    }
};

bool intervalLess(Interval &a, Interval &b){
    return a.start < b.start;
}

vector<Interval>::iterator partion(vector<Interval> &intervals, vector<Interval>::iterator left, vector<Interval>::iterator right){
    
    bool leftMove = false;
    auto reference = *left;
    
    while (left < right) {
        if (leftMove) {
            if (left->start > reference.start) {
                leftMove = false;
                *right = *left;
                right--;
            }else{
                left++;
            }
        }else{
            if (right->start <= reference.start) {
                leftMove = true;
                
                *left = *right;
                
                left++;
            }else{
                right--;
            }
        }
    }
    
    *left = reference;
    
    return left;
}

void sortIntervals(vector<Interval> &intervals, vector<Interval>::iterator left, vector<Interval>::iterator right){
    if (left >= right) {
        return;
    }
    vector<Interval>::iterator mid = partion(intervals, left, right);
    
    sortIntervals(intervals, left, mid-1);
    sortIntervals(intervals, mid+1, right);
}

vector<Interval> merge(vector<Interval> &intervals) {
    if (intervals.empty()) {
        return intervals;
    }
    //先按start排序
    sortIntervals(intervals, intervals.begin(), intervals.end()-1);
    
    auto lastIter = intervals.begin();
    for (auto iter = intervals.begin()+1; iter != intervals.end(); iter++) {
        
        if (lastIter->end >= iter->start) {
            iter->start = lastIter->start;
            iter->end = max(iter->end, lastIter->end);
            iter = intervals.erase(lastIter);
        }
        
        lastIter = iter;
    }
    
    return intervals;
}

typedef struct{
    TreeNode *node;
    int left;
    int right;
}TreeNodeRange;
TreeNode * sortedArrayToBST(vector<int> &A) {
    if (A.empty()) {
        return nullptr;
    }
    vector<TreeNodeRange>parentsRight;
    TreeNode *root = new TreeNode(A[(A.size()-1)/2]);
    TreeNodeRange cur{root, 0, (int)A.size()-1};
    
    
    while (1) {
        
        int mid = (cur.left + cur.right)/2;
        int left = (cur.left + mid-1)/2;
        int right = (cur.right+mid+1)/2;
        
        
        if (mid > cur.left) {
            TreeNode *nodeLeft = new TreeNode(A[left]);
            cur.node->left = nodeLeft;
        }
        if (mid < cur.right) {
            TreeNode *nodeRight = new TreeNode(A[right]);
            cur.node->right = nodeRight;
        }

        if (cur.node->left) {
            if (cur.node->right) {
                parentsRight.push_back({cur.node->right, mid+1, cur.right});
            }
            cur = {cur.node->left, cur.left, mid-1};
            
        }else if (cur.node->right){
            cur = {cur.node->right, mid+1, cur.right};
        }else{
            if (parentsRight.empty()) {
                break;
            }
            cur = parentsRight.back();
            parentsRight.pop_back();
        }
    }
    
    return root;
}

string addBinary(string &a, string &b) {
    auto aLen = a.length(), bLen = b.length();
    auto len = max(aLen, bLen);
    char sum[len+1];
    short carry = 0;
    for (int i = 1; i<=len; i++) {
        char aNum = 0, bNum = 0;
        if (aLen >= i) {
            aNum = a[aLen-i];
        }
        if (bLen >= i) {
            bNum = b[bLen-i];
        }
        char sum1 = aNum + bNum + carry - '0';
        if (sum1 > '1') {  //对于其他进制的计算，区别只是在这里，也就是进位的数
            sum1 -= 2;
            carry = 1;
        }
        sum[len-i+1] = sum1;
    }
    if (carry) {
        sum[0] = '1';
        return sum;
    }else{
        return sum+1;
    }
}

class NestedInteger {
public:
    // Return true if this NestedInteger holds a single integer,
    // rather than a nested list.
    bool isInteger() const;
    
    // Return the single integer that this NestedInteger holds,
    // if it holds a single integer
    // The result is undefined if this NestedInteger holds a nested list
    int getInteger() const;
    
    // Return the nested list that this NestedInteger holds,
    // if it holds a nested list
    // The result is undefined if this NestedInteger holds a single integer
    const vector<NestedInteger> &getList() const;
};

//void flatten(const vector<NestedInteger> &nestedList, vector<int> &falttenList){
//    for (auto iter = nestedList.begin(); iter != nestedList.end(); iter++) {
//        if (iter->isInteger()) {
//            falttenList.push_back(iter->getInteger());
//        }else{
//            flatten(iter->getList(), falttenList);
//        }
//    }
//}
//
//vector<int> flatten(vector<NestedInteger> &nestedList) {
//    vector<int> result;
//    
//    flatten(nestedList, result);
//    
//    return result;
//}

TreeNode * cloneTree(TreeNode * root) {
    if (root == nullptr) {
        return nullptr;
    }
    TreeNode *cloneRoot = new TreeNode(root->val);
    
    TreeNode *cur = root, *cloneCur = cloneRoot;
    vector<TreeNode *>rights, cloneRights;
    while (1) {
        if (cur->left) {
            cloneCur->left = new TreeNode(cur->left->val);
            
            if (cur->right) {
                rights.push_back(cur->right);
                
                cloneCur->right = new TreeNode(cur->right->val);
                cloneRights.push_back(cloneCur->right);
            }
            
            cur = cur->left;
            cloneCur = cloneCur->left;
        }else if (cur->right){
            cloneCur->right = new TreeNode(cur->right->val);
            
            cur = cur->right;
            cloneCur = cloneCur->right;
        }else{
            if (rights.empty()) {
                break;
            }
            cur = rights.back();
            rights.pop_back();
            
            cloneCur = cloneRights.back();
            cloneRights.pop_back();
        }
    }
    
    return cloneRoot;
}

class Tower {
private:
    stack<int> disks;
    int id;
    vector<int> disksXX;
public:
    /*
     * @param i: An integer from 0 to 2
     */
    Tower(int i) {
        id = i;
    }
    
    /*
     * @param d: An integer
     * @return: nothing
     */
    bool add(int d) {
        // Add a disk into this tower
        if (!disks.empty() && disks.top() <= d) {
//            printf("Error placing disk %d", d);
            return false;
        } else {
            disks.push(d);
            disksXX.push_back(d);
        }
        return true;
    }
    
    /*
     * @param t: a tower
     * @return: nothing
     */
    bool moveTopTo(Tower &t) {
        if (disks.empty()) {
            return false;
        }
        bool canAdd = t.add(disks.top());
        if (!canAdd) {
            return false;
        }
        disks.pop();
        disksXX.pop_back();
        return true;
    }
    
    string getState(){
        if (disks.empty()) {
            return "";
        }
        string state = "";
        for (auto iter = disksXX.begin(); iter != disksXX.end(); iter++) {
            state += to_string(*iter);
        }
        return state;
    }

    //每个状态（每个tower的disks值）是一个节点，每个操作是一个连线，就可以建立一个图，而答案就是从当前节点到最终节点的路线
    //图算法或者动态规划？
    bool printAll = false;
    vector<string> appearedStates;
    bool moveDisks(int n, Tower &destination, Tower &buffer, string &path) {
        stack<int> disk1, disk2, disk3;
        
        disk1 = getDisks();
        disk2 = destination.getDisks();
        disk3 = buffer.getDisks();
        
        auto state = getState() +"|"+ destination.getState() + "|"+ buffer.getState();
        
        if (getState().empty() && buffer.getState().empty()) {
            printf("成功!");
            return true;
        }
        
        if (find(appearedStates.begin(), appearedStates.end(), state) != appearedStates.end()) {
            if(printAll) printf("重复!\n");
            return false;
        }
        
        cout<<"\n-------------------\n"<<state<<endl;
        
        appearedStates.push_back(state);
        
        for (int i = 0; i<6; i++) {
            bool firstSucceed = false;
            string subPath;
            if (!tryMoveDisks(n, destination, buffer, i, &firstSucceed, subPath)) {
                if (firstSucceed) {
                    setDisks(disk1);
                    destination.setDisks(disk2);
                    buffer.setDisks(disk3);
                }
            }else{
                path += subPath;
                return true;
            }
        }
        

        
        return false;
    }
    
    bool tryMoveDisks(int n, Tower &destination, Tower &buffer, int operation, bool *firstSucceed, string &path){
//        printf("op:%d ",operation);
        bool moveSucceed = false;
        string opString;
        switch (operation) {
            case 0:
                moveSucceed = moveTopTo(destination);
                opString = "/1->2";
                break;
            case 1:
                moveSucceed = moveTopTo(buffer);
                opString = "/1->3";
                break;
            case 2:
                moveSucceed = destination.moveTopTo(buffer);
                opString = "/2->3";
                break;
            case 3:
                moveSucceed = destination.moveTopTo(*this);
                opString = "/2->1";
                break;
            case 4:
                moveSucceed = buffer.moveTopTo(destination);
                opString = "/3->2";
                break;
            case 5:
                moveSucceed = buffer.moveTopTo(*this);
                opString = "/3->1";
                break;
            default:
                break;
        }
        *firstSucceed = moveSucceed;
        if (!moveSucceed) {
            return false;
        }
        path.append(opString);
        
        return moveDisks(n, destination, buffer, path);
    }
    
    /*
     * @return: Disks
     */
    stack<int> getDisks() {
        // write your code here
        return disks;
    }
    
    void setDisks(stack<int> _disks){
        disks = _disks;
        
        disksXX.erase(disksXX.begin(), disksXX.end());
        while (!_disks.empty()) {
            disksXX.insert(disksXX.begin(),_disks.top());
            _disks.pop();
        }
    }
    
    friend ostream& operator<<(ostream& os, Tower& t){
        if (t.disksXX.empty()) {
            return os;
        }
        for (auto iter = t.disksXX.begin(); iter != t.disksXX.end(); ++iter) {
            os<<*iter<<" ";
        }
        return os;
    }
};

class Tower2 {
private:
    stack<int> disks;
public:
    /*
     * @param i: An integer from 0 to 2
     */
    Tower2(int i) {
        // create three towers
    }
    
    /*
     * @param d: An integer
     * @return: nothing
     */
    void add(int d) {
        // Add a disk into this tower
        if (!disks.empty() && disks.top() <= d) {
            printf("Error placing disk %d", d);
        } else {
            disks.push(d);
        }
    }
    
    /*
     * @param t: a tower
     * @return: nothing
     */
    void moveTopTo(Tower2 &t) {
        if (disks.empty()) {
            return;
        }
        t.add(disks.top());
        disks.pop();
    }
    
    /*
     * @param n: An integer
     * @param destination: a tower
     * @param buffer: a tower
     * @return: nothing
     */
    void moveDisks(int n, Tower2 &destination, Tower2 &buffer) {
        if (n == 1) {
            moveTopTo(destination);
            return;
        }
        moveDisks(n-1, buffer, destination);
        moveTopTo(destination);
        buffer.moveDisks(n-1, destination, *this);
    }
    
    /*
     * @return: Disks
     */
    stack<int> getDisks() {
        // write your code here
        return disks;
    }
    
    friend ostream& operator<<(ostream& os, Tower2& t){
        while (!t.disks.empty()) {
            os<<t.disks.top()<<" ";
            t.disks.pop();
        }
        return os;
    }
};



long long permutationIndex(vector<int> &A) {

    map<long,long> lessCounts;
    for (long i = 0; i < A.size(); i++) {
        long less = 0;
        for (long j = i; j < A.size(); j++) {
            if (A[j] < A[i]) {
                less++;
            }
        }
        
        lessCounts[A[i]] = less;
    }
    
    long long sum = 0;
    long long permutationCount = 1;
    for (long i = (long)A.size()-1; i >= 0; i--) {
        int num = A[i];
        sum += (permutationCount * lessCounts[num]);
        permutationCount *= A.size() - i;
    }
    
    return sum+1;
}

vector<int> printZMatrix(vector<vector<int>> &matrix) {
    
    bool rightTop = true;
    int row = 0, column = 0;
    vector<int>result;
    
    if (!matrix.empty() && !matrix[0].empty()) {
        result.push_back(matrix[0][0]);
    }else{
        return result;
    }
    
    auto rowCount = matrix.size(), columnCount = matrix[0].size();
    
    while (1) {
        
        if (rightTop) { //右上，不能考虑右，再考虑下
            if (column+1 < columnCount) {
                if (row-1 >= 0) {
                    result.push_back(matrix[--row][++column]);
                }else{
                    result.push_back(matrix[row][++column]);
                    rightTop = false;
                }
                
            }else if(row+1 < rowCount){
                result.push_back(matrix[++row][column]);
                rightTop = false;
            }else{
                break;
            }
        }else{                      //左下，不能考虑下，再考虑右
            if (row+1 < rowCount) {
                if (column-1 >= 0) {
                    result.push_back(matrix[++row][--column]);
                }else{
                    result.push_back(matrix[++row][column]);
                    rightTop = true;
                }
            }else if (column+1 < columnCount){
                result.push_back(matrix[row][++column]);
                rightTop = true;
            }else{
                break;
            }
        }
    }
    
    return result;
}

ListNode * addLists(ListNode * l1, ListNode * l2) {
    ListNode *result = nullptr;
    ListNode *cur1 = l1, *cur2 = l2, *curSum = nullptr;
    
    int carry = 0;
    while (cur1 != nullptr || cur2 != nullptr) {
        auto val1 = 0, val2 = 0;
        if (cur1) {
            val1 = cur1->val;
            cur1 = cur1->next;
        }
        if (cur2) {
            val2 = cur2->val;
            cur2 = cur2->next;
        }
        
        auto sum = val1 + val2 + carry;
        if (sum > 9) {
            sum -= 10;
            carry = 1;
        }else{
            carry = 0;
        }
        
        if (curSum) {
            curSum->next = new ListNode(sum);
            curSum = curSum->next;
        }else{
            curSum = result = new ListNode(sum);
        }
    }
    if (carry) {
        curSum->next = new ListNode(1);
    }
    
    return result;
}

ListNode * nthToLast(ListNode * head, int n) {
    int size = 0;
    ListNode *right = head, *left = head;
    while (right) {
        right = right->next;
        size++;
        if (size > n) {
            left = left->next;
        }
    }
    
    return left;
}

ListNode * mergeTwoLists(ListNode * l1, ListNode * l2) {
    if (l1 == nullptr) {
        return l2;
    }else if (l2 == nullptr){
        return l1;
    }
    
    ListNode *result = nullptr;
    ListNode *cur1 = l1, *cur2 = l2, *curResult = nullptr;
    while (cur1 != nullptr && cur2 != nullptr) {
        
        ListNode *less = nullptr;
        if (cur1->val < cur2->val) {
            less = cur1;
            cur1 = cur1->next;
        }else{
            less = cur2;
            cur2 = cur2->next;
        }
        
        
        if (!curResult) {
            result = less;
            curResult = less;
        }else{
            curResult->next = less;
            curResult = curResult->next;
        }
    }
    
    if (cur1) {
        curResult->next = cur1;
    }else if (cur2){
        curResult->next = cur2;
    }
    
    return result;
}

bool anagram(string s, string t) {
    if (s.length() != t.length()) {
        return false;
    }
    
    map<char, int> charTime;
    for (int i = 0; i<s.length(); i++) {
        charTime[s[i]] = charTime[s[i]] + 1;
    }
    for (int i = 0; i<t.length(); i++) {
        charTime[t[i]] = charTime[t[i]] - 1;
    }
    for (int i = 0; i<charTime.size(); i++) {
        if (charTime[i] != 0) {
            return false;
        }
    }
    
    return true;
}

bool isUnique(string &str) {
    bool charTime[256];
    memset(charTime, 0, sizeof(charTime));
    
    for (auto iter = str.begin(); iter != str.end(); iter++) {
        if (charTime[*iter]) {
            return false;
        }else{
            charTime[*iter] = true;
        }
    }
    
    return true;
}

//要返回索引，而且一大一小，这个不简单。1.不能直接重拍原数据 2.两个数相同的情况，索引会取错。
vector<int> twoSum(vector<int> &numbers, int target) {
//    quickSort(numbers);
    vector<int> sorted = numbers;
    sort(sorted.begin(), sorted.end());
    int left = 0, right = (int)sorted.size()-1;
    while (left < right) {
        int sum = sorted[left] +sorted[right];
        if (sum < target) {
            left++;
        }else if (sum > target){
            right--;
        }else{
            break;
        }
    }
    
    int index1 = 0,index2 = 0;
    for (int i = 0; i<numbers.size(); i++) {
        if (numbers[i] == sorted[left]) {
            index1 = i+1;
            break;
        }
    }
    for (int i = 0; i<numbers.size(); i++) {
        if (numbers[i] == sorted[right] && i != index1-1) {
            index2 = i+1;
            break;
        }
    }
    
    return {min(index1, index2), max(index1, index2)};
}

bool compareStrings(string &A, string &B) {
    int charTime[26];
    memset(charTime, 0, sizeof(charTime));
    for (auto iter = A.begin(); iter != A.end(); iter++) {
        charTime[*iter-'A'] = charTime[*iter-'A'] + 1;
    }
    
    for (auto iter = B.begin(); iter != B.end(); iter++) {
        charTime[*iter-'A'] = charTime[*iter-'A'] - 1;
    }
    
    for (int i = 0; i<26; i++) {
        if (charTime[i] < 0) {
            return false;
        }
    }
    
    return true;
}

struct Point {
    int x;
    int y;
    Point() : x(0), y(0) {}
    Point(int a, int b) : x(a), y(b) {}
};

struct DisPoint{
    int dis;
    Point *p;
    bool operator<(const DisPoint& other) const{
        
        if(dis != other.dis){
            return dis < other.dis;
        }
        
        if(p->x != other.p->x){
            return p->x < other.p->x;
        }
        
        if(p->y != other.p->y){
            return p->y < other.p->y;
        }
        return 0;
    }
    
    friend std::ostream& operator<<(std::ostream& os, DisPoint &disP){
        os<<"("<<disP.p->x<<","<<disP.p->y<<")"<<disP.dis<<" ";
        return os;
    }
};

int disPointCompare(DisPoint a, DisPoint b){
    if(a.dis != b.dis){
        return a.dis < b.dis ? -1:1;
    }
    
    if(a.p->x != b.p->x){
        return a.p->x < b.p->x ? -1:1;
    }
    
    if(a.p->y != b.p->y){
        return a.p->y < b.p->y ? -1:1;
    }
    return 0;
}

vector<Point> kClosest(vector<Point> points, Point origin, int k) {
    vector<DisPoint> disPoints;
    for (auto iter = points.begin(); iter != points.end(); iter++) {
        
        int dis = (iter->x-origin.x)*(iter->x-origin.x)+(iter->y-origin.y)*(iter->y-origin.y);
        
        disPoints.push_back({dis, &(*iter)});
    }
    
    sort(disPoints.begin(), disPoints.end(), disPointCompare);
    
    vector<Point> result;
    for (int i = 0; i<k; i++) {
        result.push_back(*(disPoints[i].p));
    }
    
    return result;
}

vector<Point> kClosest2(vector<Point> points, Point origin, int k){
    TFDataStruct::heap<DisPoint> heap(disPointCompare, k);
    
    for (auto iter = points.begin(); iter != points.end(); iter++) {
        
        int dis = (iter->x-origin.x)*(iter->x-origin.x)+(iter->y-origin.y)*(iter->y-origin.y);
        
        heap.append({dis, &(*iter)});
    }
    
    cout<<heap<<endl;
    
    vector<Point> result;
    for (int i = 0; i<k; i++) {
        result.push_back(*(heap.popTop().p));
    }
    
    return result;
}

int triangleCount(vector<int> &S) {
    sort(S.begin(), S.end());
    
    int count = 0;
    
    for (size_t i = 0; i<S.size()-2; i++) {
        auto lastIndex = i+2;
        for (size_t j = i+1; j < S.size()-1; j++) {
            auto left = lastIndex, right = S.size()-1;
            auto target = S[i]+S[j];
            
            //找到最大的长边，也是满足条件的最后一个;循环不变体是：left值<target<=right值
            if (target <= S[left]) {
                lastIndex = j+2;
                continue;
            }else if (target > S[right]){
                lastIndex = right;
                count += right - j; //[j+1,right]
            }else{
                while (left < right-1) {
                    auto mid = (left+right)/2;
                    if (S[mid] < target) {
                        left = mid;
                    }else if (S[mid] > target){
                        right = mid;
                    }else{
                        count += mid-1 - j; //[j+1, mid-1]
                        lastIndex = mid-1;
                        break;
                    }
                }
                
                if (left == right-1 && S[left] < target) {
                    count += left - j; //[j+1, left]
                    lastIndex = left-1;
                }
            }
        }
    }
    
    return count;
}

int digitCounts(int k, int n) {
    
    int stan = n/10, carry = 1, rest = 0;
    int digit = n % 10;
    int count = 0;
    
    while (n > 0) {
        
        int nextRest = digit*carry + rest;
        
        int left = stan - nextRest/10;  //left是高位从0到abc-1的个数，假设数为abcd
        if (k == 0) {                   //高位不能为0，所以减掉一组
            left -= carry;
        }
        if (digit > k) {
            count += left + carry;
            
        }else if (digit == k){
            count += left + rest + 1;
        }else{
            count += left;
        }
        
//        printf("%d %d %d\n",left, carry, rest);
        
        rest = nextRest;
        carry *= 10;
        n /= 10;
        digit = n % 10;
    }
    
    if (k == 0) {  //从开始计算，唯一一个不忽略的高位0
        count++;
    }
    
    return count;
}

int nthUglyNumber(int n) {
    int ugly[n];
    ugly[0] = 1;
    
    int twoIndex = 0, threeIndex = 0, fiveIndex = 0;
    int i = 1;
    while (i < n) {
        int twoNumer = ugly[twoIndex] * 2;
        int threeNumber = ugly[threeIndex] * 3;
        int fiveNumber = ugly[fiveIndex] * 5;
        
        int minNumber = min(twoNumer, min(threeNumber, fiveNumber));
        
        if (twoNumer == minNumber) {
            twoIndex++;
        }
        if (threeNumber == minNumber) {
            threeIndex++;
        }
        if (fiveNumber == minNumber) {
            fiveIndex++;
        }
        ugly[i] = minNumber;
        
//        printf("(%d)%d ",i,ugly[i]);
        i++;
        
    }
    
    return ugly[n-1];
}

int maxHeapCompare(int a, int b){
    if (a < b) {
        return -1;
    }else if (a > b){
        return 1;
    }else{
        return 0;
    }
}

int kthLargestElement(int n, vector<int> &nums) {
    
    
    //1. 顺序选，如最大值。 k*n
    //2. 顺序选，用堆管理被选的k个数,这个是方案1的升级版. lgk*n。内存问题太大。
    //3. 分治法思路，不断找一个数切分，知道区间最够小。lgn*n
    
    if (n < nums.size()/2) {  //用最小堆维持大数端
        
        TFDataStruct::heap<int> minHeap(true, n);
        
        for (int i = 0; i<nums.size(); i++) {
            if (i < n) {
                minHeap.append(nums[i]);
            }else{
                if (nums[i] >= minHeap.getTop()) {
                    minHeap.replace(nums[i], 0);
                }
            }
            
            cout<< minHeap <<endl;
        }
        
        
        
        return minHeap.getTop();
        
    }else{      //用最大堆维持小数端
        n = (int)nums.size() - n + 1;
        TFDataStruct::heap<int> maxHeap(false, n);
        
        for (int i = 0; i<nums.size(); i++) {
            if (i < n) {
                maxHeap.append(nums[i]);
            }else{
                if (nums[i] <= maxHeap.getTop()) {
                    maxHeap.replace(nums[i], 0);
                }
            }
            
            cout<< maxHeap <<endl;
        }
        
        return maxHeap.getTop();
    }
    
    
}

#pragma mark -

void printVectorSting(vector<string> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]<<" "<<endl;
    }
}

void printVectorInt(vector<int> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]<<" "<<endl;
    }
}

void printVectorStingOneLine(vector<string> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]<<" ";
    }
}

void printVectorIntOneLine(vector<int> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]<<" ";
    }
}

void printVectorNodeOneLine(vector<TreeNode *> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]->val<<" ";
    }
}


void heapTest(int startIndex, int endIndex, int testCount, long maxNumber){
    
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    double timeScale = info.numer / (double)info.denom;
    
    
    
    for (int i = startIndex; i <= endIndex; i++) {
        
        long maxSize = pow(10, i);
        
        for (int j = 0; j < testCount; j++) {
            
            int size = arc4random() % maxSize;
            
            vector<long> nums;
            nums.resize(size);
            
            for (int k = 0; k<size; k++) {
                nums.push_back(arc4random() % maxNumber);
            }
            
            printf("\n========\n");
            uint64_t time1 = mach_absolute_time();
            
            TFDataStruct::heap<long> minHeap(true,size);
            for (auto iter = nums.begin(); iter != nums.end(); iter++) {
                minHeap.append(*iter);
            }
            
//            printf("###heap created\n");
            long lastNum = 0;
            while (!minHeap.isEmpty()) {
                long num = minHeap.popTop();
                if (num < lastNum) {
                    printf("heap sort error!\n");
                }
                lastNum = num;
            }
            
            uint64_t time2 = mach_absolute_time();
            
            sort(nums.begin(), nums.end());
            uint64_t time3 = mach_absolute_time();
            
            double duration = 1e-9 * timeScale * (time2-time1);
            double duration2 = 1e-9 * timeScale * (time3-time2);
        
            printf("size:%d, maxNumber:%ld, time:%.6f, rate:%.6f, system:[%.8f, %.8f]",size, maxNumber, duration, duration/size*log2(size), duration2/duration, duration2/size*log2(size));
        }
        
    }
}

string serialize(TreeNode * root) {
    if (root == nullptr) {
        return "";
    }
    
    string result = to_string(root->val) + ",";
    
    vector<TreeNode *> *plane = new vector<TreeNode *>{root};
    while (!plane->empty()) {
        
        vector<TreeNode *> *nextPlane = new vector<TreeNode *>();
        TreeNode *cur = nullptr;
        for (auto iter = plane->begin(); iter != plane->end(); iter++) {
            
            cur = *iter;
            
            if (cur->left) {
                result.append(to_string(cur->left->val) + ",");
                nextPlane->push_back(cur->left);
            }else{
                result.append("#,");
            }
            if (cur->right) {
                result.append(to_string(cur->right->val) + ",");
                nextPlane->push_back(cur->right);
            }else{
                result.append("#,");
            }
        }
        
        free(plane);
        plane = nextPlane;
        
    }
    
    return result;
}

TreeNode * deserialize(string &data) {
    if (data.empty()) {
        return nullptr;
    }
    
    TreeNode *root = nullptr;
    
    int readCount = 1;
    int curIndex = 0;
    
    vector<TreeNode *> *plane = nullptr;
    
    do {
        
        vector<TreeNode *> *nextPlane = new vector<TreeNode *>();
        
        vector<TreeNode *>::iterator parent;
        if (plane) {
            parent = plane->begin();
        }
        bool isLeft = true;
        
        int validCount = 0;
        int i = curIndex, readed = 0;
        string number = "";
        
        while (readed < readCount) {
            if (data[i] == ',') {
                readed++;
                
                if (number != "#") {
                    validCount++;
                    
                    TreeNode *node = new TreeNode(atoi(number.c_str()));
                    nextPlane->push_back(node);
                    
                    if (root) {
                        if (isLeft) {
                            (*parent)->left = node;
                        }else{
                            (*parent)->right = node;
                        }
                    }
                    
                }
                
                isLeft = !isLeft;
                if (isLeft) {
                    parent++;
                }
                number = "";
                
            }else{
                number.push_back(data[i]);
            }
            
            i++;
        }
        
        
        
        if (root == nullptr) {
            root = nextPlane->front();
        }
        
        curIndex = i;
        readCount = 2*validCount;
        plane = nextPlane;
        
    } while (!plane->empty() && curIndex < data.size());
    
    return root;
}

TreeNode *findFirstGreater(TreeNode *root, int k){
    if (root == nullptr) {
        return nullptr;
    }
    
    if (root->val == k) {
        return root;
    }else if (root->val < k){
        return findFirstGreater(root->right, k);
    }else{
        TreeNode *leftFind = findFirstGreater(root->left, k);
        if (leftFind == nullptr) {
            return root;
        }else{
            return leftFind;
        }
    }
}

void searchRange(TreeNode * root, int k1, int k2, vector<int> &range){
    if (root == nullptr) {
        return;
    }
    
    if (root->val < k1) {
        searchRange(root->right, k1, k2, range);
    }else if (root->val > k2){
        searchRange(root->left, k1, k2, range);
    }else{
        searchRange(root->left, k1, k2, range);
        range.push_back(root->val);
        searchRange(root->right, k1, k2, range);
    }
}

vector<int> searchRange(TreeNode * root, int k1, int k2) {
    vector<int> result;
    
    searchRange(root, k1, k2, result);
    
    return result;
}

//TODO: 栈的内存堆积
vector<vector<int>> permute(vector<int> &nums, vector<int>::iterator begin) {
    
    vector<vector<int>> result;
    
    if (begin == nums.end()-1) {
        result.push_back({*begin});
        return result;
    }
    
    auto lastResult = permute(nums, begin+1);
    for (auto iter = lastResult.begin(); iter != lastResult.end(); iter++) {
        
        iter->insert(iter->begin(), *begin);
        result.push_back(*iter);
        
        for (auto iter2 = iter->begin(); iter2 != iter->end()-1; iter2++) {
            *iter2 = *(iter2+1);
            *(iter2+1)=*begin;
            
            result.push_back(*iter);
        }
    }
    
    return result;
}

vector<vector<int>> permute(vector<int> &nums) {
    if (nums.empty()) {
        
        return {{}};
    }
    return permute(nums, nums.begin());
}

//避免重复的方法，为什么会重复？两种不同的队列形式合并之后变成了同一种排列，即A+B变成E，C+D也变成E。所以放过来就是看某个排列是否有多种拆分方式。因为只考虑拆分成2部分，那么某一边唯一了，那么整个拆分也就唯一了，所以看怎么让一边一定是唯一的。
//对于有重复元素的情况，让选择唯一的方式就是把他们全部拿到同一边，因为排列组合里从n个元素里选n个只有一种方式。
//所以解法就是：把所有数分成k+1堆，这k堆是每堆里都是相同的数，且数量大于1，剩下的1堆是互不相同的“杂数”。这样每一堆之间互相融合都不会产生重复，而且最后一堆的存在，可以提高重复数比较少时的性能，它是按照无重复的情况处理的。
//融合的方式就是:想象一队队伍和迎面的另一队队伍碰面，然后我们从侧面看过去的样子。
vector<vector<int>> *partQueue(vector<int> &nums, bool *hasUnique){
    
}

void mergeTwoParts(vector<vector<int>> *permuted, vector<vector<int>> &part1, vector<vector<int>> &part2){
    
}

vector<vector<int>> permuteUnique(vector<int> &nums) {
    bool hasUnique = false;
    vector<vector<int>> *pureParts = partQueue(nums, &hasUnique);
    
    vector<vector<int>> *curPermuted = new vector<vector<int>>();
    auto pureSize = pureParts->size() - (hasUnique ? 1:0);
    
    if (pureSize > 0) {
        curPermuted->push_back(pureParts->front());
    }
    
    for (int i = 1; i<pureSize; i++) {
        mergeTwoParts(<#vector<vector<int> > &permuted#>, <#vector<vector<int> > &part1#>, <#vector<vector<int> > &part2#>)
    }
}



int main(int argc, const char * argv[]) {
    
    vector<int> nums = {};
    auto result = permute(nums);
    
    for (auto iter = result.begin(); iter != result.end(); iter++) {
        printVectorIntOneLine(*iter);
        printf("\n---------------\n");
    }
    
    return 0;
}
