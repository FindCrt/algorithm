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
#include <unordered_set>

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

//考虑空格
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

ListNode * mergeTwoLists(ListNode *h1, ListNode *t1, ListNode *h2, ListNode *t2) {
    if (h1 == t1) {
        return h2;
    }else if (h2 == t2){
        return h1;
    }
    
    ListNode *result = nullptr;
    ListNode *cur1 = h1, *cur2 = h2, *curResult = nullptr;
    while (cur1 != t1 && cur2 != t2) {
        
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
    
    if (cur1 != t1) {
        curResult->next = cur1;
    }else if (cur2 != t2){
        curResult->next = cur2;
    }
    
    return result;
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

#pragma mark - 调试函数

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
        cout<<vector[i]<<",";
    }
    cout<<endl;
}

void printVectorNodeOneLine(vector<TreeNode *> &vector){
    for (int i = 0; i<vector.size(); i++) {
        cout<<vector[i]->val<<" ";
    }
}

void printTwoDVector(vector<vector<int>> & twoDVector){
    for (auto iter = twoDVector.begin(); iter != twoDVector.end(); iter++) {
        printVectorIntOneLine(*iter);
    }
}

#pragma mark -


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
//vector<vector<int>> *partQueue(vector<int> &nums, bool *hasUnique){
//    
//}
//
//void mergeTwoParts(vector<vector<int>> *permuted, vector<vector<int>> &part1, vector<vector<int>> &part2){
//    
//}
//
//vector<vector<int>> permuteUnique(vector<int> &nums) {
//    bool hasUnique = false;
//    vector<vector<int>> *pureParts = partQueue(nums, &hasUnique);
//    
//    vector<vector<int>> *curPermuted = new vector<vector<int>>();
//    auto pureSize = pureParts->size() - (hasUnique ? 1:0);
//    
//    if (pureSize > 0) {
//        curPermuted->push_back(pureParts->front());
//    }
//    
//    for (int i = 1; i<pureSize; i++) {
//        mergeTwoParts(<#vector<vector<int> > &permuted#>, <#vector<vector<int> > &part1#>, <#vector<vector<int> > &part2#>)
//    }
//}

vector<vector<int>> subsets(vector<int> &nums, vector<int>::iterator start){
    if (start == nums.end()) {
        return {{}};
    }
    auto subSets = subsets(nums, start+1);
    
    auto size = subSets.size();
    for (int i = 0; i<size; i++) {
        
        auto oneSet = subSets[i];
        oneSet.insert(oneSet.begin(), *start);
        printVectorIntOneLine(oneSet);
        subSets.push_back(oneSet);
    }
    
    printTwoDVector(subSets);
    
    return subSets;
}

vector<vector<int>> subsets(vector<int> &nums) {
    if (nums.empty()) {
        return {{}};
    }
    
    return subsets(nums, nums.begin());
}

vector<vector<int>> subsetsWithDup(vector<int> &nums, vector<int>::iterator start) {
    if (start == nums.end()) {
        return {{}};
    }
    
    int dupCount = 1;
    auto nextIter = start+1;
    while (nextIter != nums.end() && *nextIter == *start) {
        dupCount++;
        nextIter++;
    }
    
    auto subSets = subsetsWithDup(nums, nextIter);
    
    auto size = subSets.size();
    for (int i = 0; i<size; i++) {
        for (int j = 0; j<dupCount; j++) {
            
            vector<int> oneSet;
            if (j == 0) {
                oneSet = subSets[i];
                
            }else{
                oneSet = subSets.back();
            }
            
            oneSet.insert(oneSet.begin(), *start);
            subSets.push_back(oneSet);
        }
    }
    
    return subSets;
}

vector<vector<int>> subsetsWithDup(vector<int> &nums) {
    if (nums.empty()) {
        return {{}};
    }
    sort(nums.begin(), nums.end());
    return subsetsWithDup(nums, nums.begin());
}

bool isInterleave(string &s1, int i1, string &s2, int i2, string &s3, int i3){
    if (i1 == s1.size()) {
        return s2.substr(i2, s2.size()-i2).compare(s3.substr(i3, s3.size()-i3)) == 0;
    }else if (i2 == s2.size()){
        return s1.substr(i1, s1.size()-i1).compare(s3.substr(i3, s3.size()-i3)) == 0;
    }
    
    char h1 = s1[i1];
    char h2 = s2[i2];
    char h3 = s3[i3];
    
    if (h3 == h2) {
        if (h3 == h1) {
            return isInterleave(s1, i1+1, s2, i2, s3, i3+1) || isInterleave(s1, i1, s2, i2+1, s3, i3+1);
        }else{
            return isInterleave(s1, i1, s2, i2+1, s3, i3+1);
        }
    }else if (h3 == h1){
        return isInterleave(s1, i1+1, s2, i2, s3, i3+1);
    }else{
        return false;
    }
}

bool isInterleave(string &s1, string &s2, string &s3) {
    return isInterleave(s1, 0, s2, 0, s3, 0);
}



int partitionArray(vector<int> &nums, int k) {
    if (nums.empty()) {
        return 0;
    }
    int left = 0, right = (int)nums.size()-1;
    
    while (1) {
        while (nums[left] < k) {
            left++;
            
            if (left > right) {
                return left;
            }
        }
        
        while (nums[right] >= k) {
            right--;
            
            if (left > right) {
                return left;
            }
        }
        
        auto temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
        
        if (left == right-1) {
            return right;
        }
        
        left++;
        right--;
        
//        printf("[%d, %d]\n",left,right);
    }
}

struct QueenSolution {
    vector<Point> points;
    vector<int> validX;
    vector<int> validY;
    vector<int> invalidDiff1; //x-y
    vector<int> invalidDiff2; //x+y
    
    QueenSolution(QueenSolution *other){
        this->points = other->points;
        this->validX = other->validX;
        this->validY = other->validY;
        this->invalidDiff1 = other->invalidDiff1;
        this->invalidDiff2 = other->invalidDiff2;
    }
    
    QueenSolution(){};
};

vector<QueenSolution*> solveNQueens(int size, int count){
    
    if (size < count || size < 1 || count < 0) {
        return {};
    }
    
    if (size == 1) {
        
        if (count == 1) {
            QueenSolution *solution = new QueenSolution();
            solution->points = {Point(0, 0)};
            solution->validX = {};
            solution->validY = {};
            solution->invalidDiff1 = {0};
            solution->invalidDiff2 = {0};
            
            return {solution};
        }else{
            QueenSolution *solution = new QueenSolution();
            solution->points = {};
            solution->validX = {0};
            solution->validY = {0};
            solution->invalidDiff1 = {};
            solution->invalidDiff2 = {};
            
            return {solution};
        }
    }
    
    vector<QueenSolution*> result;
    
    //subCount * 1
    auto subRes0 = solveNQueens(size-1, count);
    for (auto iter = subRes0.begin(); iter != subRes0.end(); iter++) {
        QueenSolution *solution = *iter;
        solution->validX.push_back(size-1);
        solution->validY.push_back(size-1);
        
        result.push_back(solution);
    }
    
    
    //subCount
    auto subRes1 = solveNQueens(size-1, count-1);
    for (auto iter = subRes1.begin(); iter != subRes1.end(); iter++) {
        QueenSolution *solution = *iter;
        
        for (int i = 0; i < solution->validX.size(); i++) {
            
            auto x = solution->validX[i];
            auto diff1 = x - size +1, diff2 = x + size - 1;
            bool unused1 = find(solution->invalidDiff1.begin(), solution->invalidDiff1.end(), diff1) == solution->invalidDiff1.end();
            bool unused2 = find(solution->invalidDiff2.begin(), solution->invalidDiff2.end(), diff2) == solution->invalidDiff2.end();
            
            if (unused1 && unused2) {
                QueenSolution *copy = new QueenSolution(solution);
                copy->points.push_back(Point(x, size-1));
                copy->validX.erase(copy->validX.begin()+i);
                copy->validX.push_back(size-1);
                copy->invalidDiff1.push_back(diff1);
                copy->invalidDiff2.push_back(diff2);
                
                result.push_back(copy);
            }
        }
        
        for (int i = 0; i<solution->validY.size(); i++) {
            
            auto y = solution->validY[i];
            auto diff1 = size-1 - y, diff2 = size - 1 + y;
            
            bool unused1 = find(solution->invalidDiff1.begin(), solution->invalidDiff1.end(), diff1) == solution->invalidDiff1.end();
            bool unused2 = find(solution->invalidDiff2.begin(), solution->invalidDiff2.end(), diff2) == solution->invalidDiff2.end();
            
            if (unused1 && unused2) {
                QueenSolution *copy = new QueenSolution(solution);
                copy->points.push_back(Point(size-1, y));
                copy->validY.erase(copy->validY.begin()+i);
                copy->validY.push_back(size-1);
                copy->invalidDiff1.push_back(diff1);
                copy->invalidDiff2.push_back(diff2);
                
                result.push_back(copy);
            }
        }
        
        if (find(solution->invalidDiff1.begin(), solution->invalidDiff1.end(), 0) == solution->invalidDiff1.end()) {
            QueenSolution *copy = new QueenSolution(solution);
            copy->points.push_back(Point(size-1, size-1));
            copy->invalidDiff1.push_back(0);
            copy->invalidDiff2.push_back(2*size-2);
            
            result.push_back(copy);
        }
        
        free(solution);
    }
    
    //[x * y]
    auto subRes2 = solveNQueens(size-1, count-2);
    for (auto iter = subRes2.begin(); iter != subRes2.end(); iter++) {
        QueenSolution *solution = *iter;
        
        vector<int> possibleX,possibleY;
        for (int i = 0; i < solution->validX.size(); i++) {
            
            auto x = solution->validX[i];
            auto diff1 = x - size +1, diff2 = x + size - 1;
            bool unused1 = find(solution->invalidDiff1.begin(), solution->invalidDiff1.end(), diff1) == solution->invalidDiff1.end();
            bool unused2 = find(solution->invalidDiff2.begin(), solution->invalidDiff2.end(), diff2) == solution->invalidDiff2.end();
            
            if (unused1 && unused2) {
                possibleX.push_back(x);
            }
        }
        
        for (int i = 0; i < solution->validY.size(); i++) {
            
            auto y = solution->validY[i];
            
            auto diff1 = size-1 - y, diff2 = size - 1 + y;
            bool unused1 = find(solution->invalidDiff1.begin(), solution->invalidDiff1.end(), diff1) == solution->invalidDiff1.end();
            bool unused2 = find(solution->invalidDiff2.begin(), solution->invalidDiff2.end(), diff2) == solution->invalidDiff2.end();
            
            
            if (unused1 && unused2) {
                possibleY.push_back(y);
            }
        }
        
        for (int i = 0; i < possibleX.size(); i++) {
            for (int j = 0; j<possibleY.size(); j++) {
                if (possibleX[i] != possibleY[j]) {
                    QueenSolution *copy = new QueenSolution(solution);
                    
                    copy->points.push_back(Point(possibleX[i], size-1));
                    copy->validX.erase(find(copy->validX.begin(), copy->validX.end(), possibleX[i]));
                    
                    copy->invalidDiff1.push_back(possibleX[i]-size+1);
                    copy->invalidDiff2.push_back(possibleX[i]+size-1);
                    
                    
                    copy->points.push_back(Point(size-1, possibleY[j]));
                    copy->validY.erase(find(copy->validY.begin(), copy->validY.end(), possibleY[j]));

                    copy->invalidDiff1.push_back(size-1-possibleY[j]);
                    copy->invalidDiff2.push_back(size-1+possibleY[j]);
                    
                    result.push_back(copy);
                }
            }
        }
        
        free(solution);
    }
    
    return result;
}

vector<vector<string>> solveNQueens(int n) {
    
    auto solutions = solveNQueens(n, n);
    vector<vector<string>> result;
    
    for (int i = 0; i<solutions.size(); i++) {
        QueenSolution *solution = solutions[i];
        
        vector<string> str_solution;
        for (int j = 0; j < n; j++) {
            str_solution.push_back(string(n, '.'));
        }
        
        for (int j = 0; j<solution->points.size(); j++) {
            Point p = solution->points[j];
            str_solution[p.x][p.y] = 'Q';
        }
        
        result.push_back(str_solution);
    }
    
    return result;
}

#pragma mark - queen

int maxxx = 11;
int queen[12];
int sum=0; /* max为棋盘最大坐标 */

void show() /* 输出所有皇后的坐标 */
{
    int i;
    printf("(");
    //i代表行数，queen[i]代表当前行元素所处的列数，
    //注意此处下标是从0开始的
    
    for(i = 0; i < maxxx; i++)
    {
        printf(" %d", queen[i]+1);
    }
    printf(")\n");
    //每次输出一种解的时候，那么他的解的数量就会增加1
    sum++;
}

//此函数用于判断皇后当前皇后是否可以放在此位置
int PLACE(int n) /* 检查当前列能否放置皇后 */
{
    //queen[i] == queen[n]用于保证元素不能再同一列
    //abs(queen[i] - queen[n]) == abs(n - i)用于约束元素不能再同一行且不能再同一条斜线上
    int i;
    for(i = 0; i < n; i++) /* 检查横排和对角线上是否可以放置皇后 */
    {
        if(queen[i] == queen[n] || abs(queen[i] - queen[n]) == abs(n - i))
        {
            return 0;
        }
    }
    return 1;
}

//核心函数，回溯法的思想
void NQUEENS(int n) /* 回溯尝试皇后位置,n为横坐标 */
{
    int i;
    for(i = 0; i < maxxx; i++)
    {
        //首先将皇后放在第0列的位置，对于第一次来说是肯定成立的
        //所以第一次将皇后放在第0行0列的位置
        queen[n] = i; /* 将皇后摆到当前循环到的位置 */
        if(PLACE(n))
        {
            if(n == maxxx - 1)
            {
                show(); /* 如果全部摆好，则输出所有皇后的坐标 */
            }
            else
            {
                NQUEENS(n + 1); /* 否则继续摆放下一个皇后 */
            }
        }
    }
}

int maxSubArray(vector<int> &nums, int start, int end){
    
    if (start == end) {
        return nums[end];
    }
    
    int start_max = nums[start];
    int curSum = start_max;
    
    for (int i = start+1; i<=end; i++) {
        curSum += nums[i];
        start_max = max(start_max, curSum);
    }
    
    return max(start_max, maxSubArray(nums, start+1, end));
}

//只考虑包含end的
int end_subs(vector<int> &nums, int end, int *marks, bool isMin){
    
    if (end < 0) {
        return 0;
    }
    
    int result = nums[end];
    
    int subResult = end_subs(nums, end-1, marks, isMin);
    if (isMin) {
        if (subResult < 0) { //求最小值时，下一级的解是小于0的，加上才会更小
            result += subResult;
        }
    }else{
        if (subResult > 0) {
            result += subResult;
        }
    }
    
    marks[end] = result;
    
    return result;
}

int start_subs(vector<int> &nums, int start, int *marks, bool isMin){
    
    if (start >= nums.size()) {
        return 0;
    }
    
    int result = nums[start];
    
    int subResult = start_subs(nums, start+1, marks, isMin);
    if (isMin) {
        if (subResult < 0) { //求最小值时，下一级的解是小于0的，加上才会更小
            result += subResult;
        }
    }else{
        if (subResult > 0) {
            result += subResult;
        }
    }
    
    marks[start] = result;
    
    return result;
}

//选择一个点，分割成两个区域,分别取最大值和最小值。关键点在于，解的两个区域一定是相邻的，这样就可以只考虑包含边界的子区间，直接从O(n^2)降到了O(n)
int maxDiffSubArrays(vector<int> &nums) {
    
    if (nums.empty()) {
        return 0;
    }
    
    int start_minSum[nums.size()], start_maxSum[nums.size()], end_minSum[nums.size()], end_maxSum[nums.size()];;
    memset(start_minSum, 0, sizeof(int)*nums.size());
    memset(start_maxSum, 0, sizeof(int)*nums.size());
    memset(end_minSum, 0, sizeof(int)*nums.size());
    memset(end_maxSum, 0, sizeof(int)*nums.size());
    
    start_subs(nums, 0, start_minSum, true);
    start_subs(nums, 0, start_maxSum, false);
    
    end_subs(nums, (int)nums.size()-1, end_minSum, true);
    end_subs(nums, (int)nums.size()-1, end_maxSum, false);
    
    int maxDiff = 0;
    
    for (int i = 0; i<nums.size()-1; i++) {
        int startMin = start_minSum[i+1];
        int endMax = end_maxSum[i];
        
        int startMax = start_maxSum[i+1];
        int endMin = end_minSum[i];

        if (i == 0) {
            maxDiff = max(abs(endMax-startMin), abs(startMax-endMin));
        }else{
            maxDiff = max(max(abs(endMax-startMin), abs(startMax-endMin)), maxDiff);
        }
    }
    
    return maxDiff;
}

void sortLetters(string &chars) {
    if (chars.empty()) {
        return;
    }
    
    char part = 'a';
    
    size_t i = 0, j = chars.size()-1;
    
    
    while (i < j) {
        while (chars[i] >= part) {
            i++;
            if (i == j) {
                return;
            }
        }
        while (chars[j] < part) {
            j--;
            if (i == j) {
                return;
            }
        }
        
        auto temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
        
        i++;
        j--;
    }
}

bool comparexx(const int &a,const int &b)
{
    return a>b;
}

vector<int> previousPermuation(vector<int> &nums) {
    
    //只记录索引最小的那次，因为有高位时一定有低位可以交换，而这时一定低位，所以高位无用。
    map<int,long> numIndexes;
    
    int markCount = 0;
    for (long i = (long)nums.size()-1; i >= 0; i--) {
        int num = nums[i];
        if (numIndexes.find(num) == numIndexes.end()) {
            numIndexes[num] = i;
            
            markCount++;
            if (markCount == 10) {
                break;
            }
        }
    }
    
    //低位的交换一定小于高位的交换,即使是考虑低位重排后
    for (long start = (long)nums.size()-2; start >= 0; start--) {
        int num = nums[start];
        
        //优先数字，而不是位置
        for (int i = num-1; i>= 0; i--) {
            auto index = numIndexes[i];
            if (index > start) {
                nums[start] = nums[index];
                nums[index] = num;
                
                sort(nums.begin()+start+1, nums.end(), comparexx);
                
                return nums;
            }
        }
    }
    
    //没有最小的了，取最大的排列
    sort(nums.begin(), nums.end(), comparexx);
    
    return nums;
}

vector<int> nextPermutation(vector<int> &nums) {
    
    if (nums.empty()) {
        return nums;
    }
    
    //只记录索引最小的那次，因为有高位时一定有低位可以交换，而这时一定低位，所以高位无用。
    map<int,long> numIndexes;
    
    int maxNum = nums.front();
    
    int markCount = 0;
    for (long i = (long)nums.size()-1; i >= 0; i--) {
        int num = nums[i];
        
        if (numIndexes.find(num) == numIndexes.end()) {
            numIndexes[num] = i;
            
            markCount++;
            if (markCount == 10) {
                break;
            }
        }
        
        if (num > maxNum) maxNum = num;
    }
    
    //低位的交换一定小于高位的交换,即使是考虑低位重排后
    for (long start = (long)nums.size()-2; start >= 0; start--) {
        int num = nums[start];
        
        //优先数字，而不是位置
        for (int i = num+1; i<= maxNum; i++) {
            auto index = numIndexes[i];
            if (index > start) {
                nums[start] = nums[index];
                nums[index] = num;
                
                sort(nums.begin()+start+1, nums.end());
                
                return nums;
            }
        }
    }
    
    //没有更大的了，取最小的排列
    sort(nums.begin(), nums.end());
    
    return nums;
}

vector<vector<int>> twoSum(vector<int> &numbers, int sum, int start, int end){
    
    vector<vector<int>> result;
    
    int left = start, right = end;
    
    while (left < right) {
        
        int curSum = numbers[left]+numbers[right];
        
        if (curSum < sum) {
            left++;
        }else if (curSum > sum){
            right--;
        }else{
            if (result.empty() || numbers[left] != result.back().front()) {  //avoid duplicate
                result.push_back({numbers[left], numbers[right]});
            }
            
            left++;
            right--;
        }
    }
    
    return result;
}

vector<vector<int>> threeSum(vector<int> &numbers, int target = 0) {
    sort(numbers.begin(), numbers.end());
    
    vector<vector<int>> result;
    int repeatCount = 1;
    for (int i = 0; i<numbers.size(); i+= repeatCount) {
        int first = numbers[i];
        
        repeatCount = 1;
        while (i+repeatCount < numbers.size() && numbers[i+repeatCount] == first) {
            repeatCount++;
        }
        
        //two first, find another num to make sum 0.
        if (repeatCount > 1) {
            int subTarget = -2*first;
            
            if (find(numbers.begin()+i+repeatCount, numbers.end(), subTarget) != numbers.end()) {
                result.push_back({first, first, subTarget});
            }
        }
        
        //one first, find other two num
        int subTarget = -1*first;
        auto subResults = twoSum(numbers, subTarget, i+repeatCount, (int)numbers.size()-1);
        for (int k = 0; k < subResults.size(); k++) {
            
            auto solution = subResults[k];
            solution.insert(solution.begin(), first);
            
            result.push_back(solution);
        }
        
        //three first
        if (first == 0 && repeatCount > 2) {
            result.push_back({0, 0, 0});
        }
    }
    
    return result;
}

vector<vector<int>> kSum(vector<int> &numbers, int k ,int target, int start, map<int, int> &numCounts){
    
    if (k == 2) {
        return twoSum(numbers, target, start, (int)numbers.size()-1);
    }else if (k == 1){
        if (find(numbers.begin()+start, numbers.end(), target) != numbers.end()) {
            return {{target}};
        }
        return {};
    }

    vector<vector<int>> result;
    
    int repeatCount = 1;
    for (int i = start; i<numbers.size(); i+=repeatCount) {
        int first = numbers[i];
        repeatCount = numCounts[first];
        
        //每个循环代表，取j第一个数
        for (int j = 1; j <= min(repeatCount, k); j++) {
            
            int subTarget = target - j*first;
            
            if(subTarget == 0 && j == k){
                result.push_back(vector<int>(k, first));
            }else{
                //获取剩余的部分的解，然后拼接
                auto subResults = kSum(numbers, k-j, subTarget, i+repeatCount, numCounts);
                
                for (int m = 0; m < subResults.size(); m++) {
                    auto subRes = subResults[m];
                    for (int l = 0; l<j; l++) {
                        subRes.insert(subRes.begin(), first);
                    }
                    result.push_back(subRes);
                }
            }
            
            
        }
    }
    
    return result;
}

vector<vector<int>> kSum(vector<int> &numbers, int k ,int target){
    
    if (numbers.empty()) {
        return {};
    }
    
    sort(numbers.begin(), numbers.end());
    
    map<int, int> numCounts;
    for (int i = 0; i<numbers.size(); i++) {
        
        int num = numbers[i];
        if (numCounts.find(num) == numCounts.end()) {
            numCounts[num] = 1;
        }else{
            numCounts[num] = numCounts[num]+1;
        }
    }
    
    return kSum(numbers, k, target, 0, numCounts);
}


vector<vector<int>> fourSum(vector<int> &numbers, int target) {
    return kSum(numbers, 4, target);
}

int closestNum(vector<int> &numbers, int target, int start){
    size_t left = start, right = numbers.size()-1;
    
    if (numbers[left] >= target) {
        return numbers[left];
    }else if (numbers[right] <= target){
        return numbers[right];
    }
    
    while (left < right-1) {
        auto mid = (left+right)/2;
        if (numbers[mid] < target) {
            left = mid;
        }else if (numbers[mid] > target){
            right = mid;
        }else{
            return target;
        }
    }
    
    if (abs(numbers[left]-target) < abs(numbers[right]-target)) {
        return numbers[left];
    }else{
        return numbers[right];
    }
}

int twoSumClosest(vector<int> &numbers, int target, int start){
    size_t left = start, right = numbers.size()-1;
    
    int diff = INT_MAX;
    
    while (left < right) {
        
        int curSum = numbers[left]+numbers[right];
        
        if (abs(curSum-target) < abs(diff)) {
            diff = curSum-target;
        }
        
        if (curSum < target) {
            left++;
        }else if (curSum > target){
            right--;
        }else{
            return target;
        }
    }
    
    return diff+target;
}

int threeSumClosest(vector<int> &numbers, int target) {
    sort(numbers.begin(), numbers.end());
    
    int diff = INT_MAX;
    
    int repeatCount = 1;
    for (int i = 0; i<numbers.size(); i+= repeatCount) {
        int first = numbers[i];
        
        repeatCount = 1;
        while (i+repeatCount < numbers.size() && numbers[i+repeatCount] == first) {
            repeatCount++;
        }
        
        if (repeatCount > 2) {
            if (abs(3*first-target) < diff) {
                diff = 3*first-target;
            }
        }
        
        //two first, find another num to make sum 0.
        if (repeatCount > 1) {
            int subTarget = target-2*first;
            
            int otherNum = closestNum(numbers, subTarget, i+repeatCount);
            if (abs(otherNum-subTarget) < abs(diff)) {
                diff = otherNum-subTarget;
            }
        }
        
        //one first, find other two num
        int subTarget = target-1*first;
        int twoSum = twoSumClosest(numbers, subTarget, i+repeatCount);
        if (abs(twoSum-subTarget) < abs(diff)) {
            diff = twoSum-subTarget;
        }
    }
    
    return diff + target;
}

vector<int> searchRange(vector<int> &A, int target) {
    if (A.empty()) {
        return {-1, -1};
    }
    
    int find = -1;
    int left1 = -1, right1 = -1, left2 = -1, right2 = (int)A.size();
    
    int i = -1, j = (int)A.size();
    while (i < j-1) {
        auto mid = (i+j)/2;
        if (A[mid] < target) {
            i = mid;
        }else if (A[mid] > target){
            j = mid;
        }else{
            find = mid;
            break;
        }
    }
    
    if (find < 0) {
        return {-1, -1};
    }else{
        right1 = left2 = find;
    }
    
    while (left1 < right1-1) {
        auto mid = (left1+right1)/2;
        if (A[mid] < target) {
            left1 = mid;
        }else{
            right1 = mid;
        }
    }
    
    while (left2 < right2-1) {
        auto mid = (left2+right2)/2;
        if (A[mid] > target) {
            right2 = mid;
        }else{
            left2 = mid;
        }
    }
    
    return {right1, left2};
}

int search(vector<int> &A, int target) {
    
    if (A.empty()) {
        return -1;
    }
    
    int part = A[0];
    
    int i = -1, j = (int)A.size();
    while (i < j-1) {
        auto mid = (i+j)/2;
        if (A[mid] < part) {
            j = mid;
        }else if (A[mid] >= part){
            i = mid;
        }
    }
    
    int reverse = i;
    if (A[reverse] == target) {
        return reverse;
    }
    
    int size = (int)A.size();
    i = reverse, j = reverse+size+1;
    while (i < j-1) {
        auto mid = (i+j)/2;
        
        int num = 0;
        if (mid >= size) {
            num = A[mid-size];
        }else{
            num = A[mid];
        }
        if (num < target) {
            i = mid;
        }else if (num > target){
            j = mid;
        }else{
            return mid >= size ? mid-size:mid;
        }
    }
    
    return -1;
}

vector<vector<int>> zigzagLevelOrder(TreeNode * root) {
    
    if (root == nullptr) {
        return {};
    }
    
    vector<vector<int>> result;
    
    vector<TreeNode *> *curNodes = new vector<TreeNode *>(), *nextNodes = nullptr;
    curNodes->push_back(root);
    
    bool fromLeft = true;
    
    while (!curNodes->empty()) {
        
        nextNodes = new vector<TreeNode *>();
        vector<int> oneResult;
        
        for (auto iter = curNodes->end()-1; iter != curNodes->begin()-1; iter--) {
            auto node = *iter;
            oneResult.push_back(node->val);
            
            if (fromLeft) {
                if (node->left) nextNodes->push_back(node->left);
                if (node->right) nextNodes->push_back(node->right);
            }else{
                if (node->right) nextNodes->push_back(node->right);
                if (node->left) nextNodes->push_back(node->left);
            }
        }
        
        fromLeft = !fromLeft;
        
        free(curNodes);
        curNodes = nextNodes;
        nextNodes = nullptr;
        
        result.push_back(oneResult);
    }
    
    return result;
}

class SVNRepo {
    public:
    static bool isBadVersion(int k){
        return k >= 2147483647;
    };
};

int findFirstBadVersion(int n) {
    int i = 0, j = n;
    while (i < j-1) {
        int mid = i+(j-i)/2;
        if (SVNRepo::isBadVersion(mid)) {
            j = mid;
        }else{
            i = mid;
        }
    }
    
    return j;
}

inline static int slope(vector<int> &A, int index){
    int incre1 = A[index] - A[index-1];
    int incre2 = A[index+1] - A[index];
    
    if (incre1 > 0) {
        if (incre2 > 0) {
            return 1;  //increase
        }else{
            return 0;  //peek
        }
    }else{
        if (incre2 > 0) {
            return -1; //valley
        }else{
            return -2; //decrease
        }
    }
}

//因为是离散的，会出错：{1, 2, 1, 3, 4, 5, 5, 4}.其实4-5-5-4之间是有一个峰值的，但是离散的看不出来，或说没有一个绝对的峰值点，两个5都是峰值。  *** 相邻数字不同的条件下才正确 ***

//如果画一条斜率的图，那么开始是正的，结束时负的，而我们要找的就是斜率为0的位置，用二分法就可以。
//避免找到低谷，要确保左边是正的，右边是负的。
int findPeak(vector<int>& A) {
    
    int i = 0, j = (int)A.size()-1;
    while (i < j-1) {
        int mid = i+(j-i)/2;
        
        int slope_mid = slope(A, mid);
        if (slope_mid < 0) {
            j = mid;
        }else if (slope_mid > 0){
            i = mid;
        }else{
            return mid;
        }
    }
    
    return 0;
}

string longestCommonPrefix(vector<string> &strs) {
    
    if (strs.empty()) {
        return "";
    }
    
    string result = "";
    
    string minLen = strs.front();
    
    for (int i = 0; i<minLen.size(); i++) {
        char checker = minLen[i];
        
        for (int j = 0; j<strs.size(); j++) {
            if (strs[j][i] != checker) {
                return result;
            }
        }
        
        result.push_back(checker);
    }
    
    return result;
}

int longestCommonSubstring(string &A, string &B) {
    map<int, vector<int>> charIndexB;
    
    for (int i = 0; i<B.size(); i++) {
        int index = B[i];
        
        if (charIndexB.find(index) != charIndexB.end()) {
            charIndexB[index].push_back(i);
        }else{
            charIndexB[index] = {i};
        }
    }
    
    int maxLen = 0;
    
    for (int i = 0; i<A.size(); i++) {
        char first = A[i];
        
        auto pairIndexes = charIndexB[first];
        for (int j = 0; j<pairIndexes.size(); j++) {
            int offset = pairIndexes[j] - i;
            
            int len = 1;
            while (A[i+len] == B[i+len+offset] && i+len < A.size() && i+len+offset < B.size()) {
                len++;
            }
            
            if (len > maxLen) {
                maxLen = len;
            }
        }
    }
    
    return maxLen;
}

int singleNumberII(vector<int> &A) {
    map<int, int> counts;
    
    for (int i = 0; i<A.size(); i++) {
        
        int num = A[i];
        if (counts.find(num) == counts.end()) {
            counts[num] = 1;
        }else{
            if (counts[num] == 2) {
                counts.erase(num);
            }else{
                counts[num] = counts[num]+1;
            }
        }
    }
    
    return (*counts.begin()).first;
}

vector<int> singleNumberIII(vector<int> &A) {
    map<int, bool> counts;
    
    for (int i = 0; i<A.size(); i++) {
        int num = A[i];
        
        if (counts.find(num) == counts.end()) {
            counts[num] = true;
        }else{
            counts.erase(num);
        }
    }
    
    return {counts.begin()->first, (++counts.begin())->first};
}

int MinAdjustmentCost(vector<int> &A, int target) {
    
    if (A.empty()) {
        return 0;
    }
    
    int maxNum = 100;
    int costs[A.size()][maxNum];
    
    for (int i = 0; i<maxNum; i++) {
        costs[0][i] = abs(A[0] - i);
    }
    
    for (int i = 1; i<A.size(); i++) {
        for (int j = 0; j<maxNum; j++) {
            int minLastCost = INT_MAX;
            for (int k = max(j-target, 0); k<=min(j+target, maxNum); k++) {
                minLastCost = min(costs[i-1][k], minLastCost);
            }
            
            costs[i][j] = minLastCost + abs(A[i]-j);
        }
    }
    
    int minCost = costs[A.size()-1][0];
    for (int i = 1; i<maxNum; i++) {
        minCost = min(minCost, costs[A.size()-1][i]);
    }
    
    return minCost;
}

int backPack(int m, vector<int> &A) {
    int residue[m+1][A.size()+1];
    
    for (int i = 0; i<=m; i++) {
        residue[i][0] = i;
    }
    
    for (int count = 1; count <= A.size(); count++) {
        for (int resM = 0; resM<=m; resM++) {
            int current = A[A.size()-count];
            
            int unpackStart = residue[resM][count-1];
            if (resM >= current) {
                int packStart = residue[resM-current][count-1];
                residue[resM][count] = min(unpackStart, packStart);
            }else{
                residue[resM][count] = unpackStart;
            }
        }
    }
    
    return m - residue[m][A.size()];
}
    
//因为二叉树的性质，这个路线一定是，有一个点上升，然后下降，最多只有一个拐弯处，因为取了子节点就没法回头了。所以就变成了由某个节点发出的两条路线和最大值，而二叉树每个节点最多两个子节点，所以还不用选。
//单边的路线可以丢弃了一边路线的
void maxPathSum(TreeNode * root, int *maxPath, int *curMaxSum){
    
    int leftMaxPath = 0, rightMaxPath = 0;
    if (root->left) {
        maxPathSum(root->left, &leftMaxPath, curMaxSum);
        if (leftMaxPath < 0) { //可以不要
            leftMaxPath = 0;
        }
    }
    if (root->right) {
        maxPathSum(root->right, &rightMaxPath, curMaxSum);
        if (rightMaxPath < 0) {
            rightMaxPath = 0;
        }
    }
    
    *curMaxSum = max(*curMaxSum, leftMaxPath+rightMaxPath+root->val);
    
    *maxPath = max(leftMaxPath, rightMaxPath)+root->val;
}

int maxPathSum(TreeNode * root) {
    int maxPath = INT_MIN, result = INT_MIN;
    maxPathSum(root, &maxPath, &result);
    
    return result;
}

bool isValidBST(TreeNode * root, int *minNum, int *maxNum){
    if (root == nullptr) {
        return true;
    }
    
    if (root->left) {
        
        int leftMin = 0,leftMax = 0;
        bool leftValid = isValidBST(root->left, &leftMin, &leftMax);
        if (!leftValid) {
            return false;
        }else if (leftMax >= root->val){
            return false;
        }
        
        *minNum = leftMin;
    }else{
        *minNum = root->val;
    }
    
    if (root->right) {
        
        int rightMin = 0, rightMax = 0;
        bool rightValid = isValidBST(root->right, &rightMin, &rightMax);
        if (!rightValid) {
            return false;
        }else if (rightMin <= root->val){
            return false;
        }
        
        *maxNum = rightMax;
    }else{
        *maxNum = root->val;
    }
    
    return true;
}

bool isValidBST(TreeNode * root) {
    int min = 0, max = 0;
    return isValidBST(root, &min, &max);
}

//tail是不包含的
ListNode *list_partion(ListNode *head, ListNode *tail){
    ListNode *slow = head, *fast = head->next;
    
    auto contrast = head->val;
    
    while (fast != tail) {
        if (fast->val < contrast) {
            
            slow = slow->next;
            
            auto temp = slow->val;
            slow->val = fast->val;
            fast->val = temp;
        }
        
        fast = fast->next;
    }
    
    auto temp = slow->val;
    slow->val = contrast;
    head->val = temp;
    
    return slow;
}

ListNode *list_quickSort(ListNode * head, ListNode *tail){
    
    if (head == nullptr) {
        return head;
    }
    
    ListNode *partion = list_partion(head, tail);
    
    if (partion != head) {
        list_quickSort(head, partion);
    }
    if (partion->next != tail) {
        list_quickSort(partion->next, tail);
    }
    
    return head;
}

//tail不包含
ListNode *list_getMid(ListNode * head){
    ListNode *slow = head, *fast = head;
    
    while (fast != nullptr && fast->next != nullptr) {
        fast = fast->next;
        if (fast) {
            fast = fast->next;
        }
        
        slow = slow->next;
    }
    
    return slow;
}

//tail不包含
ListNode *list_mergeSort(ListNode * head){
    
    if (head->next == nullptr) {
        return head;
    }
    
    auto mid = list_getMid(head);
    
    
    auto right = list_mergeSort(mid->next);
    mid->next = nullptr;
    auto left = list_mergeSort(head);
    
    return mergeTwoLists(left, right);
}

ListNode * sortList(ListNode * head) {
//    return list_quickSort(head, nullptr);
    return list_mergeSort(head);
}

ListNode *reverseList(ListNode *list, ListNode *startPre = nullptr, ListNode *end = nullptr){
    if (list == nullptr) {
        return nullptr;
    }
    
    ListNode *start = nullptr;
    
    if (startPre == nullptr) {
        start = list;
    }else{
        start = startPre->next;
        if (start == nullptr) {
            return list;
        }
    }
    
    ListNode *revHead = start, *revTail = start;
    
    auto next = revTail->next;
    
    while (next != end) {
        revTail->next = next->next;
        next->next = revHead;
        
        revHead = next;
        
        next = revTail->next;
    }
    
    if (startPre == nullptr) {
        return revHead;
    }else{
        startPre->next = revHead;
        return list;
    }
}

void reorderList(ListNode * head) {
    if (head == nullptr) {
        return;
    }
    ListNode *mid = list_getMid(head);
    ListNode *normalHead = reverseList(head, mid);
    
    ListNode *revHead = mid->next;
    mid->next = nullptr;
    
    while (revHead) {
        auto next = revHead->next;
        
        revHead->next = normalHead->next;
        normalHead->next = revHead;
        
        normalHead = normalHead->next->next;
        revHead = next;
    }
}

//不能标记，因为你不知道从哪个节点开始进入了环。快慢节点相当于造了一个动态的环。
//假设第一个环的长度是T，那么快节点走到2T的时候，慢节点走到T,它们刚好相遇。一定有解，说明一定会相遇。
bool hasCycle(ListNode * head) {
    ListNode *fast = head, *slow = head;
    
    while (fast != nullptr) {
        fast = fast->next;
        if (fast == nullptr) {
            return false;
        }
        fast = fast->next;
        
        slow = slow->next;
        
        if (slow == fast) {
            return true;
        }
    }
    
    return false;
}

static inline int listNodeMinHeapCompare(ListNode *a, ListNode *b){
    if (a->val < b->val) {
        return -1;
    }else if (a->val > b->val){
        return 1;
    }else{
        return 0;
    }
}

ListNode *mergeKLists(vector<ListNode *> &lists) {
    TFDataStruct::heap<ListNode *> headMinHeap(listNodeMinHeapCompare, lists.size());
    
    for (int i = 0; i<lists.size(); i++) {
        ListNode *head = lists[i];
        if (head) {
            headMinHeap.append(head);
        }
    }
    
    cout<<headMinHeap<<endl;
    
    ListNode *head = nullptr, *tail = nullptr;
    
    while (!headMinHeap.isEmpty()) {
        ListNode *cur = headMinHeap.popTop();
        
        if (head == nullptr) {
            head = tail = cur;
        }else{
            tail->next = cur;
            tail = cur;
        }
        
        if (cur->next) headMinHeap.append(cur->next);
    }
    
    return head;
}

struct RandomListNode {
    int label;
    RandomListNode *next, *random;
    RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
};

RandomListNode *copyRandomList(RandomListNode *head) {
    if (head == nullptr) {
        return nullptr;
    }
    
    RandomListNode *cur = head;
    RandomListNode *newHead = nullptr, *newLast = nullptr;
    
    while (cur) {
        
        RandomListNode *newCur = new RandomListNode(cur->label);
        newCur->random = cur->random;
        
        if (newHead == nullptr) {
            newHead = newCur;
            newLast = newCur;
        }else{
            newLast->next = newCur;
            newLast = newCur;
        }
        
        auto next = cur->next;
        cur->next = newCur;
        
        cur = next;
    }
    
    auto newCur = newHead;
    while (newCur) {
        if (newCur->random) {
            newCur->random = newCur->random->next;
        }
        
        newCur = newCur->next;
    }
    
    return newHead;
}

bool wordBreak(string &s, unordered_set<string> &dict) {
    bool match[s.size()+1];
    memset(match, 0, sizeof(match));
    
    match[0] = true;
    
    
    for (size_t i = 1; i<s.size()+1; i++) {
        for (auto iter = dict.begin(); iter != dict.end(); iter++) {
            auto len = iter->size();
            
            if (len > i) {
                continue;
            }
            
            if (s.substr(i-len, len).compare(*iter) == 0 && match[i-len]) {
                match[i] = true;
            }
        }
    }
    
    return match[s.size()];
}

//简单回文判断
inline bool isPalindrome2(string &s, size_t start, size_t end){
    size_t i = 0;
    while (i < (end-start+1)/2) {
        if (s[i+start] != s[end-i]) {
            return false;
        }
        
        i++;
    }
    
    return true;
}

int minRoute(bool **routes, size_t start, size_t end, map<size_t, int> &save){
    
    if (save.find(start) != save.end()) {
        return save[start];
    }
    
    if (start > end) {
        return 0;
    }else if (start == end){
        return 1;
    }
    
    int subMin = INT_MAX;
    for (auto i = start; i<=end; i++) {
        if (routes[start][i]) {
            subMin = min(subMin, minRoute(routes, i+1, end, save));
        }

    }
    
    save[start] = subMin+1;
    
    return subMin+1;
}

int minCut(string &s) {
    
    bool *routes[s.size()];
    for (int i = 0; i<s.size(); i++) {
        routes[i] = new bool[s.size()];
        memset(routes[i], 0, sizeof(bool)*s.size());
    }
    
    //初始化回文标记，因为回文的性质，固定一个中心，从内向外扩散判断，某个长度不是回文，那么之后都不是了。
    //复杂度n^2,每一对有且仅有一次判断。
    for (size_t sum = 0; sum <= 2*s.size()-2; sum++) {
        long left = sum/2, right = sum-left;
        while (left>= 0 && right <= s.size()) {
            if (s[left] == s[right]) {
                routes[left][right] = true;
            }else{
                break;
            }
            
            left--;
            right++;
        }
    }
    
    map<long, unsigned long> save;
    save[s.size()-1] = 1;
    
    for (long i = s.size()-2; i>= 0; i--) {
        
        unsigned long curMin = s.size();
        if (routes[i][s.size()-1]) {
            curMin = 1;
        }else{
            for (auto j = i; j<s.size(); j++) {
                if (routes[i][j]) {
                    curMin = min(curMin, save[j+1]+1);
                }
            }
        }
        
        save[i] = curMin;
    }
    
    return (int)save[0]-1;
}

//重复的保留一个
ListNode * deleteDuplicates(ListNode * head) {
    if (head == nullptr) {
        return head;
    }
    
    map<int, bool> exist;
    ListNode *cur = head, *next = cur->next;
    exist[cur->val] = true;
    
    while (next) {
        if (exist.find(next->val) != exist.end()) {
            cur->next = next->next;
        }else{
            exist[next->val] = true;
            cur = next;
        }
        
        next = cur->next;
    }
    
    return head;
}

//重复的全删掉
ListNode * deleteDuplicates2(ListNode * head) {
    if (head == nullptr) {
        return head;
    }
    
    map<int, bool> exist;
    ListNode *cur = head, *next = cur->next;
    exist[cur->val] = false;
    
    while (next) {
        if (exist.find(next->val) != exist.end()) {
            cur->next = next->next;
            exist[next->val] = true; //more than 1 times or Duplicate
        }else{
            exist[next->val] = false; //just exist
            cur = next;
        }
        
        next = cur->next;
    }
    
    cur = head;
    ListNode *pre = nullptr;
    while (cur) {
        if (exist[cur->val]) { //Duplicate
            if (pre) {
                pre->next = cur->next;
            }else{
                head = cur->next;
            }
            
        }else{
            pre = cur;
        }
        
        cur = cur->next;
    }
    
    return head;
}

bool canJump2(vector<int> &A) {
    if (A.empty()) {
        return true;
    }
    
    bool canJList[A.size()];
    memset(canJList, 0, sizeof(canJList));
    canJList[0] = true;
    
    for (int i = 1; i<A.size(); i++) {
        for (int j = 0; j<i; j++) {
            if (canJList[j] && A[j]+j >= i) {
                canJList[i] = true;
                break;
            }
        }
    }
    
    return canJList[A.size()-1];
}

bool canJump(vector<int> &A) {
    if (A.empty()) {
        return true;
    }
    
    size_t cur = 0, next = A[0];
    while (next < A.size()-1) {
        
        auto maxJump = next;
        
        for (auto i = cur+1; i<=next; i++) {
            maxJump = max(A[i]+i, maxJump);
        }
        
        if (maxJump == next) { //下一轮没有实现跨越，就会被锁死
            return false;
        }
        
        cur = next;
        next = maxJump;
    }
    
    return true;
}

/**
 *  动态规划的核心是找到从k-1到k之间的过渡模式，从简到难有3种：
 *  1. 直接去掉一个元素 
 *  2. 所有[0, k-1] 共同影响k
 *  3. 符合某些条件的[0, k-1]，这题就是这样，条件是：可以一步跳跃到k的低级元素。有些二维的动态规划也是这样，下面一题的编辑距离就是这类型的经典。
 */
int jump2(vector<int> &A) {
    int step[A.size()];
    memset(step, 0, sizeof(step));
    
    for (int i = 1; i<A.size(); i++) {
        int minStep = i; //最多为i-1,i是不可能的值
        
        for (int j = 0; j<i; j++) {
            if (A[j]+j >= i) { //可以一步跳过来
                minStep = min(minStep, step[j]);
            }
        }
        
        if (minStep < i) {
            step[i] = minStep+1;
        }else{
            return -1;  //没有解
        }

    }
    
    return step[A.size()-1];
}

int jump(vector<int> &A) {
    int step[A.size()];
    memset(step, 0, sizeof(step));
    
    for (int i = 0; i<A.size(); i++) {
        
        for (int j = i+1; j<=min(A[i]+i, (int)A.size()); j++) {
            if (step[j] == 0 || step[i]+1 < step[j]) {
                
                step[j] = step[i]+1;
            }
        }
    }
    
    return step[A.size()-1];
}

//因为i情况下的结果只需要知道i-1的结果便可以求出，所以原本的二维表可以缩减为一维表，dp就是当前i所对应的那一列的记录。
int numDistinct(string &S, string &T) {
    // write your code here
    vector<int> dp(T.length()+1);
    dp[0] = 1;
    for(int i=1;i<=S.length();i++)
    {
        for(int j=T.length();j>0;j--)
            dp[j] += T[j-1]==S[i-1]?dp[j-1]:0;
    }
    return dp[T.length()];
}

inline void fillDis(string &word1, string &word2, int x, int y, int **dis){
    
    if (x == word1.size()) {
        dis[x][y] = word2.size() - y;
        return;
    }else if(y == word2.size()){
        dis[x][y] = word1.size() - x;
        return;
    }
    
    if (word1[x] == word2[y]) {
        dis[x][y] = dis[x+1][y+1];  //相等，不改
    }else{
        //依次是替换、删除和增加
        dis[x][y] = min(dis[x+1][y+1],min(dis[x+1][y], dis[x][y+1]))+1;
    }
}

//119. 编辑距离
int minDistance(string &word1, string &word2) {
    int *dis[word1.size()+1]; //按2维数组方式，x的长度代表指针的个数，x方向采用word1
    for (int i = 0; i<=word1.size(); i++) {
        dis[i] = new int[word2.size()+1];
        memset(dis[i], 0, sizeof(int)*(word2.size()+1));
    }
    
    int minWidth = (int)min(word1.size(), word2.size());
    
    for (int w = 0; w <= minWidth; w++) {
        int x = word1.size()-w, y = word2.size()-w;  //x方向采用word1，跟上面匹配
        
        fillDis(word1, word2, x, y, dis);
        
        for (int v = y-1; v >= 0; v--) {
            fillDis(word1, word2, x, v, dis);
        }
        
        for (int h = x-1; h >= 0; h--) {
            fillDis(word1, word2, h, y, dis);
        }
    }
    
//    for (int i = 0; i<=word1.size(); i++) {
//        for (int j = 0; j<=word2.size(); j++) {
//            printf("%d ",dis[i][j]);
//        }
//        
//        printf("\n");
//    }
    
    return dis[0][0];
}

template <class T>
class LadderGraphNode{
public:
    T val;
    vector<LadderGraphNode*> links;
//    LadderGraphNode *shortestPre;
    int step = -1;
    
    LadderGraphNode(T val):val(val){};
};

inline bool oneModify(string &str1, string &str2){
    bool modify = false;
    for (int i = 0; i<str1.size(); i++) {
        if (str1[i] != str2[i]) {
            
            if (modify) {
                return false;
            }else{
                modify = true;
            }
        }
    }
    return true;
}

//120. 单词接龙
//想象每个单词是一个球，如果两个单词只需要经过一次替换就会相等，那么在它们之间连上一条线。start和end也加入到这个词典里，然后整体就会形成一个网，一个立体的网。题目的需求就转变为了：寻找从start到end的一条最短路径。
int ladderLength(string &start, string &end, unordered_set<string> &dict) {
    
    if (start.compare(end) == 0) {
        return 1;
    }
    if (oneModify(start, end)) {
        return 2;
    }
    
    vector<LadderGraphNode<string>*> nodes;
    for (auto iter = dict.begin(); iter != dict.end(); iter++) {
        
        nodes.push_back(new LadderGraphNode<string>(*iter));
    }
    
    for (int i = 0; i<nodes.size(); i++) {
        auto cur = nodes[i];
        
        for (int j = i+1; j<nodes.size(); j++) {
            auto other = nodes[j];
            if (oneModify(cur->val, other->val)) {
                cur->links.push_back(other);
                other->links.push_back(cur);
            }
        }
    }
    
    auto startNode = new LadderGraphNode<string>(start);
    startNode->step = 1;
    auto endNode = new LadderGraphNode<string>(end);
    
    for (int i = 0; i<dict.size(); i++) {
        
        auto cur = nodes[i];
        if (oneModify(startNode->val, cur->val)) {
            startNode->links.push_back(cur);
//            cur->links.push_back(startNode);
        }
        if (oneModify(endNode->val, cur->val)) {
            cur->links.push_back(endNode);
//            endNode->links.push_back(cur);
        }
    }
    
    auto border = new  vector<LadderGraphNode<string>*>();
    border->push_back(startNode);
    
    while (!border->empty()) {
        auto nextBorder = new  vector<LadderGraphNode<string>*>();
        
        for (auto iter = border->begin(); iter != border->end(); iter++) {
            auto node1 = *iter;
            for (auto iter2 = node1->links.begin(); iter2 != node1->links.end(); iter2++) {
                auto node2 = *iter2;
                
                if (node2 == endNode) {
                    free(border);
                    free(nextBorder);
                    
                    //逆序取出结果
//                    vector<string> result = {end};
//                    auto back = node1;
//                    while (back != startNode) {
//                        result.insert(result.begin(), back->val);
//                        back = back->shortestPre;
//                    }
//                    result.insert(result.begin(), start);
//                    printVectorSting(result);
                    
                    return node1->step+1;
                }
                
                if (node2->step < 0) {
                    node2->step = node1->step+1;
//                    node2->shortestPre = node1;
                    
                    nextBorder->push_back(node2);
                }
            }
        }
        
        free(border);
        border = nextBorder;
    }
    
    free(border);
    
    return 0;
}

//代表一个层级的分叉口，层级指距离开始的长度;
//使用这个机构就可以把多个分叉抽象成一个结构，配合深度搜索，每一刻只需要保存一个临时变量
struct CharFork{
    vector<Point> points;  //每个层级有多个分叉口
    int pos; //层级大小，也是对应字符在word里的索引
    
    CharFork(int pos):pos(pos){};
};

//检测下一步可行的位置
inline void checkNextFork(vector<vector<char>> &board, string &word, CharFork *nextFork, int x, int y, int pos, vector<vector<bool>> &found){
    if (x >= board.size() || y >= board[x].size() || found[x][y]) {
        return;
    }
    if (board[x][y] == word[pos]) {
        nextFork->points.push_back(Point(x, y));
    }
}

//123. 单词搜索
//1. 从一个图里把树提取出来 2. 一边构建一边遍历，而且使用深度搜索方式
bool exist(vector<vector<char>> &board, string &word) {
    
    if (word.empty()) {
        return false;
    }
    
    vector<CharFork*> forks;
    
    char first = word[0];
    CharFork *fork = new CharFork(0);
    
    for (int i = 0; i<board.size(); i++) {
        auto row = board[i];
        for (int j = 0; j<row.size(); j++) {
            if (row[j] == first) {
                fork->points.push_back(Point(i, j));
            }
        }
    }
    
    if (fork->points.empty()) {
        return false;
    }else if (word.size() == 1){
        return true;
    }
    
    vector<Point> route;
    vector<vector<bool>> found;
    
    for (int i = 0; i<board.size(); i++) {
        found.push_back(vector<bool>());
        for (int j = 0; j<board[i].size(); j++) {
            found[i].push_back(false);
        }
    }
    
    //curFork是当前最前面一层，forks保存了之前的多层
    CharFork *curFork = fork;
    
    while (1) {
        
        while (curFork->points.empty()) {
            free(curFork);
            if (forks.empty()) {
                return false;
            }
            curFork = forks.back();
            forks.pop_back();
            
            found[route.back().x][route.back().y] = false;
            route.pop_back();
        }

        Point curPoint = curFork->points.back();
        
        int nextPos = curFork->pos+1;
        CharFork *nextFork = new CharFork(nextPos);
        checkNextFork(board, word, nextFork, curPoint.x+1, curPoint.y, nextPos, found);
        checkNextFork(board, word, nextFork, curPoint.x-1, curPoint.y, nextPos, found);
        checkNextFork(board, word, nextFork, curPoint.x, curPoint.y+1, nextPos, found);
        checkNextFork(board, word, nextFork, curPoint.x, curPoint.y-1, nextPos, found);
        
        curFork->points.pop_back();
        
        if (!nextFork->points.empty()) {
            if (nextPos == word.size()-1) {
                return true;
            }
            //把当前的存入，切换到下一层
            forks.push_back(curFork);
            curFork = nextFork;
            
            route.push_back(curPoint); //可以继续进行，表示这条路还有可能性，才保存
            found[curPoint.x][curPoint.y] = true;
        }
    }
}

int longestConsecutive(vector<int> &num) {
    unordered_set<int> exist;
    for (auto iter = num.begin(); iter != num.end(); iter++) {
        exist.insert(*iter);
    }
    
    int maxLen = 0;
    for (auto iter = num.begin(); iter != num.end(); iter++) {
        
        int cur = *iter;
        
        if (exist.find(cur-1) != exist.end()) {
            continue;
        }
        
        int len = 0;
        
        do {
            cur++;
            len++;
        } while (exist.find(cur) != exist.end());
        
        maxLen = max(maxLen, len);
    }
    
    return maxLen;
}

//125. 背包问题 II
//递归算法，复杂度O(2^n),因为会有很多的重复,m-A[start]这里，5-3和6-4是一样的。
int backPackII2(int m, vector<int> &A, vector<int> &V, int start){
    
    if (m <= 0) {
        return 0;
    }
    if (start == A.size()-1) {
        return m >= A[start] ? V[start] : 0;
    }
    
    int save = 0;
    if (m >= A[start]){
        save = backPackII2(m-A[start], A, V, start+1)+V[start];
    }
    int unsave = backPackII2(m, A, V, start+1);
    
    return max(save, unsave);
}


//125. 背包问题 II
//只依赖上一层条件就可以往下推，所以滚动数组，每次values表示当前条件的结果值
int backPackII(int m, vector<int> &A, vector<int> &V) {
    
    int values[m+1];
    memset(values, 0, sizeof(values));
    
    //每轮存储的选择范围是[0, i]
    for (int i = 0; i < A.size(); i++) {
        
        //需要需靠上一层的低容量数据，所以低容量值不能被干扰，所以从大到小遍历
        for (int j = m; j >=0; j--) {
            if (j < A[i]) {  //装不下，值不变
                continue;
            }
            values[j] = max(values[j], values[j-A[i]]+V[i]);
        }
        
//        for (int i = 0; i<= m; i++) {
//            printf("%d ",values[i]);
//        }
//        
//        printf("\n\n--------------\n");
    }
    return values[m];
}

struct DirectedGraphNode {
    int label;
    vector<DirectedGraphNode *> neighbors;
    DirectedGraphNode(int x) : label(x) {};
};

template <class T>
struct GraphNodeFork{
public:
    int dep;
    vector<T> nodes;
    
    GraphNodeFork(int dep):dep(dep){};
};

vector<DirectedGraphNode*> topSort(vector<DirectedGraphNode*>& graph) {
    unordered_set<DirectedGraphNode*> begin;
    for (auto iter = graph.begin(); iter != graph.end(); iter++) {
        begin.insert(*iter);
    }
    
    for (auto iter = graph.begin(); iter != graph.end(); iter++) {
        
        auto node = *iter;
        for (auto iter2 = node->neighbors.begin(); iter2 != node->neighbors.end(); iter2++) {
            begin.erase(*iter2);
        }
    }
    
    //curFork是当前最前面一层，forks保存了之前的多层
    auto curFork = new GraphNodeFork<DirectedGraphNode*>(0);
    for (auto iter = begin.begin(); iter != begin.end(); iter++) {
        curFork->nodes.push_back(*iter);
    }
    
    vector<DirectedGraphNode*> route;
    vector<GraphNodeFork<DirectedGraphNode*> *> forks;
    
    while (1) {
        
        while (curFork->nodes.empty()) {
            free(curFork);
            if (forks.empty()) {  //应该不可能到这里
                return route;
            }
            curFork = forks.back();
            forks.pop_back();
        
            route.pop_back();
        }
        
        auto node = curFork->nodes.back();
        
        int nextPos = curFork->dep+1;
        auto nextFork = new GraphNodeFork<DirectedGraphNode*>(nextPos);
        for (int i = 0; i<node->neighbors.size(); i++) {
            auto nbor = node->neighbors[i];
            
            //路径里还没来得及加入自身，但自身可能在neighbors里，也要排除
            if (nbor == node) {
                return route;
            }
            if (find(route.begin(), route.end(),nbor) == route.end()) {
                nextFork->nodes.push_back(nbor);
            }
        }
        
        curFork->nodes.pop_back();
        
        if (!nextFork->nodes.empty()) {
            
            //把当前的存入，切换到下一层
            forks.push_back(curFork);
            curFork = nextFork;
            
            route.push_back(node); //可以继续进行，表示这条路还有可能性，才保存
        }else if(route.size() == graph.size()-1){
            route.push_back(node); //没有进行下去的路了，终结
            return route;
        }
    }
}

vector<DirectedGraphNode*> topSort2(vector<DirectedGraphNode*> graph) {
    vector<DirectedGraphNode*> ret;
    if(graph.empty())
        return ret;
    
    map<DirectedGraphNode*,int> in; //in为入度
    stack<DirectedGraphNode*>   s;  //保存入度为零的节点
    for(auto e:graph){
        for(auto i:e->neighbors)
            ++in[i];              //记录每个节点的入度
    }
    
    for(auto e:graph)
        if(0==in[e])
            s.push(e);         //入度为零的节点入栈
    
    while(!s.empty()){        //BFS遍历,搜寻入度为零的节点
        DirectedGraphNode* cur=s.top();
        s.pop();             //当前节点出栈时，它的相邻节点入度都减一
        ret.push_back(cur);
        for(auto e:cur->neighbors){
            if(--in[e]==0)    //减一后为零则入栈
                s.push(e);
        }
    }
    return ret;
}

vector<ListNode*> rehashing(vector<ListNode*> hashTable) {
    
    if (hashTable.empty()) {
        return hashTable;
    }
    
    long capacity = hashTable.size()*2;
    vector<ListNode *> result(capacity);
    
    for (auto iter = hashTable.begin(); iter != hashTable.end(); iter++) {
        auto cur = *iter;
        
        while (cur) {
            auto index = cur->val % capacity;
            if (index < 0) {
                index += capacity;
            }
            
            ListNode *newList = result[index];
            if (newList == nullptr) {
                result[index] = cur;
            }else{
                
                while (newList->next) { //末端保持为空
                    newList = newList->next;
                }
                newList->next = cur;
            }
            
            auto next = cur->next;
            cur->next = nullptr; //末端保持为空
            cur = next;
        }
    }
    
    return result;
}

//数字组合的树结构是一个奇葩状态，因为数可以重复使用，那么每个节点包含多种使用数量;
//不是单纯的包含子节点，而是每种不同的子节点选择，自身的贡献值也是变化的
//因为不同父节点会连接到相同子节点去，所以实际不是树而是图了。
//跟单词接龙那题结构是类似的，但是那题只需要求最小路径，所以使用广度搜索，而这里是要提取所有的完整路径，需要深度搜索
template <class T>
class CombinationNode {
public:
    //key是val的使用次数，value是使用了这么多次的val后，对应的下一个界别的解。
    map<int, CombinationNode*> childern;
    T val;
    
    CombinationNode(T val):val(val){};
    
    //把多叉树的路径全部提取出来
    vector<vector<T>> allRoutes(){
        vector<vector<T>> routes;
        
        if (childern.empty()) {
            return {{val}};
        }
        for (auto iter = childern.begin(); iter != childern.end(); iter++) {
            int num = iter->first;
            
            if (iter->second) {
                auto subRoutes = iter->second->allRoutes();
                
                for (auto iter2 = subRoutes.begin(); iter2 != subRoutes.end(); iter2++) {
                    for (int i = 0; i<num; i++) {
                        iter2->push_back(val);
                    }
                    routes.push_back(*iter2);
                }
            }else{
                //没有下一级了，路径就是自身的各种类型
                routes.push_back(vector<int>(num, val));
            }
            
        }
        
        return routes;
    }
};

//135. 数字组合 背包问题变种
//相比背包问题复杂很多，主要因为：1. 需求把解输出，而不是求一个最大值，所以必须记录每种情况，然后回溯求解  2.每个数的使用次数不受限，这样导致每个从2分支(不使用或使用0次)状态变成了n分支状态(使用n次)
vector<vector<int>> combinationSum(vector<int> &candidates, int target) {
    
    sort(candidates.begin(), candidates.end());
    
    CombinationNode<int>* flag[candidates.size()][target+1];
    for (int i = 0; i<candidates.size(); i++) {
        memset(flag[i], 0, sizeof(flag[i]));
    }
    
    //起始边界：多个目标、一个选择值，要么没有解，要么有解也只有一个解。
    int i = 0, t = 0;
    while (t <= target) {
        flag[0][t] = new CombinationNode<int>(candidates[0]);
        flag[0][t]->childern[i] = nullptr; //没有下一级了
        i++;
        t = i*candidates[0];
    }
    
    for (int i = 1; i<candidates.size(); i++) {
        for (int j = target; j>=0; j--) {
            
            auto node = new CombinationNode<int>(candidates[i]);
            
            //k是这个数取的个数，t是剩余的目标数
            int k = 0, t = j;
            while (t >= 0) {
                
                if (flag[i-1][t]) {
                    node->childern[k] = flag[i-1][t];
                }
                k++;
                t = j - candidates[i]*k;
            }
            
            if (node->childern.empty()) {  //下级无解，这一节点也就不存了
                free(node);
            }else{
                flag[i][j] = node;
            }
        }
    }
    
    //从后往前回溯找解，路径模型是一个有向图，但是是特殊的图，只有相邻的两层之间有链接，想象起来起来有点像织的布或者肌肉
    vector<vector<int>> solutions;
    
    if (!flag[candidates.size()-1][target]) {
        return solutions;
    }
    
    return flag[candidates.size()-1][target]->allRoutes();
}

template <class T>
class LinkedGraphNode {
public:
    T val;
    vector<LinkedGraphNode *> links;
    
    LinkedGraphNode(T val):val(val){};
    
    vector<vector<T>> allRoutes(LinkedGraphNode *end){
        
        if (links.empty()) {
            return {{val}};
        }
        
        vector<vector<T>> routes;
        
        for (auto iter = links.begin(); iter != links.end(); iter++) {
            auto subRoutes = iter->allRoutes();
            
            for (auto iter2 = subRoutes.begin(); iter2 != subRoutes.end(); iter2++) {
                iter2->push_back(val);
                routes.push_back(*iter2);
            }
        }
        
        return routes;
    }
};

vector<vector<string>> partition(string &s, bool **routes, int start, int end){
    
    vector<vector<string>> result;
    
    if (start == end-1) {
        return {{string(1,s.back())}}; //只有一个字符
    }
    
    for (int i = start; i<end; i++) {
        
        if (routes[start][i]) {
            
            if (i == end-1) {
                result.push_back({s.substr(start, i-start+1)});
            }else{
                auto subResult = partition(s, routes, i+1, end);
                
                for (auto iter = subResult.begin(); iter != subResult.end(); iter++) {
                    iter->insert(iter->begin(), s.substr(start, i-start+1));
                    
                    result.push_back(*iter);
                }
            }
            
        }
    }
    
    return result;
}

vector<vector<string>> partition(string &s) {
    
    bool *routes[s.size()];
    for (int i = 0; i<s.size(); i++) {
        routes[i] = new bool[s.size()];
        memset(routes[i], 0, sizeof(bool)*s.size());
    }
    
    //初始化回文标记，因为回文的性质，固定一个中心，从内向外扩散判断，某个长度不是回文，那么之后都不是了。
    //复杂度n^2,每一对有且仅有一次判断。
    for (size_t sum = 0; sum <= 2*s.size()-2; sum++) {
        long left = sum/2, right = sum-left;
        while (left>= 0 && right <= s.size()) {
            if (s[left] == s[right]) {
                routes[left][right] = true;
            }else{
                break;
            }
            
            left--;
            right++;
        }
    }
    
    return partition(s, routes, 0, (int)s.size());
}

unsigned long quickPower(int a, int b){
    unsigned long s = 1;
    while (b > 0) {
        if (b & 1) { //奇数
            s *= a;
        }
        
        a *= a;
        b >>= 1;
    }
    
    return s;
}

int fastPower(int a, int b, int n){
    int s = 1;
    while (b > 0) {
        if (b & 1) { //奇数
            s *= a%n;
            s %= n;
        }
        
        a *= a;
        b >>= 1;
    }
    
    return s;
}

void sortColors2(vector<int> &colors, int k) {
    int slot[k];
    memset(slot, 0, sizeof(slot));
    
    for (int i = 0; i<colors.size(); i++) {
        slot[colors[i]-1]++;
    }
    
    int j = 0;
    for (int i = 0; i<k; i++) {
        for (int x = 0; x<slot[i]; x++) {
            colors[j] = i+1;
            j++;
        }
    }
}

void rerange(vector<int> &A) {
    int cur = 0,pos = 0,nega=0;
    
    int positiveCount = 0;
    for (int i = 0; i<A.size(); i++) {
        if (A[i] < 0) {
            positiveCount-=1;
        }else{
            positiveCount+=1;
        }
    }
    
    //0 -1 true, 1 false
    bool positive = positiveCount <= 0;
    
    while (cur < A.size()) {
        positive = !positive;
        if (positive) {
            if (A[cur] < 0) {
                while (A[pos]<0 && pos < A.size()) {
                    pos++;
                }
                auto temp = A[cur];
                A[cur] = A[pos];
                A[pos] = temp;
            }
            pos++;
            if (nega<=cur){
                nega = cur+1;
            }
        }else{
            if (A[cur] > 0) {
                while (A[nega]>0 && nega < A.size()) {
                    nega++;
                }
                auto temp = A[cur];
                A[cur] = A[nega];
                A[nega] = temp;
            }
            
            nega++;
            if (pos<=cur) {
                pos = cur+1;
            }
        }
        
        cur++;
    }
}

void sortColors(vector<int> &nums) {
    int num1=0, num2=0, num3 = 0;
    
    for (int i = 0; i<nums.size(); i++) {
        if (nums[i] == 0) {
            num1++;
        }else if (nums[i] == 1){
            num2++;
        }else{
            num3++;
        }
    }
    
    int i = 0;
    for (int j = 0; j<num1; j++, i++) {
        nums[i] = 0;
    }
    
    for (int j = 0; j<num2; j++, i++) {
        nums[i] = 1;
    }
    
    for (int j = 0; j<num3; j++, i++) {
        nums[i] = 2;
    }
}

int maxProfit(vector<int> &diff, int start, int *maxInclude) {
    if (start == diff.size()-1) {
        *maxInclude = diff[start];
        return diff[start];
    }
    
    int includeCur = 0;
    auto subResult = maxProfit(diff, start+1, &includeCur);
    
    if (includeCur>0) {
        includeCur += diff[start];
    }else{
        includeCur = diff[start];
    }
    
//    printf("[%d] %d\n",start,includeCur);
    
    *maxInclude = includeCur;
    
    return max(subResult, includeCur);
}

int maxProfit(vector<int> &prices) {
    if (prices.size() < 2) {
        return 0;
    }
    vector<int> diff;
    
    for (int i=0; i<prices.size()-1; i++) {
        diff.push_back(prices[i+1]-prices[i]);
    }
    
//    printVectorIntOneLine(diff);
    
    int i;
    return max(maxProfit(diff, 0, &i), 0);
}

int findMin(vector<int> &nums) {
    if (nums.front() <= nums.back()) {
        return nums.front();
    }
    if (nums[nums.size()-2] > nums.back()) {
        return nums.back();
    }
    
    int left = 0, right = (int)nums.size()-2;
    int stan = nums.back();
    while (left < right-1) {
        int mid = left+(right-left)/2;
        
        if (nums[mid] < stan) {
            right = mid;
        }else{
            left = mid;
        }
    }
    
    return nums[right];
}

int findMin2(vector<int> &nums) {
    if (nums.size() == 1) {
        return nums.front();
    }
    
    //左边大于，右边小于等于
    int left = 0, right = (int)nums.size()-1;
    int stan = nums.back();
    
    while (left < right-1) {
        if (nums[left] == stan) {
            left++;
        }else{
            break;
        }
    }
    
    if (nums[left] < stan) {
        return nums[left];
    }
    
    while (left < right-1) {
        
        int mid = left+(right-left)/2;
        
        if (nums[mid] <= stan) {
            right = mid;
        }else if (nums[mid] > stan){
            left = mid;
        }
    }
    
    return nums[right];
}

void rotate(vector<vector<int>>& matrix)
{
    if (matrix.empty())
        return;
    for (int x = 0; x < (matrix.size()-1) / 2+1; ++x)
    {
        auto revx = matrix.size()-x-1;
        for (int y = 0; y < matrix.size() / 2; ++y)
        {
            auto revy = matrix.size()-y-1;
            int tmp = matrix[y][x];
            matrix[y][x] = matrix[revx][y];
            matrix[revx][y] = matrix[revy][revx];
            matrix[revy][revx] = matrix[x][revy];
            matrix[x][revy] = tmp;
        }
    }
}

void setZeroes(vector<vector<int>> &matrix) {
    if (matrix.empty()) {
        return;
    }
    
    int m = matrix.size();
    int n = matrix.front().size();
    
    bool firstHZero = false;
    for (int j = 0; j<n; j++) {
        if (!matrix[0][j]) {
            firstHZero = true;
            break;
        }
    }
    
    bool firstVZero = false;
    for (int i = 0; i<m; i++) {
        if (!matrix[i][0]) {
            firstVZero = true;
            break;
        }
    }
    
    for (int i = 1; i<m; i++) {
        for (int j = 1; j<n; j++) {
            if (!matrix[i][j]) {
                matrix[0][j] = 0;
                matrix[i][0] = 0;
            }
        }
    }
    
    for (int i = 1; i<m; i++) {
        if (!matrix[i][0]) {
            for (int j = 1; j<n; j++) {
                matrix[i][j] = 0;
            }
        }
    }
    
    for (int j = 1; j<n; j++) {
        if (!matrix[0][j]) {
            for (int i = 1; i<m; i++) {
                matrix[i][j] = 0;
            }
        }
    }
    
    if (firstHZero) {
        for (int j = 0; j<n; j++) {
            matrix[0][j] = 0;
        }
    }
    
    if (firstVZero) {
        for (int i = 0; i<m; i++) {
            matrix[i][0] = 0;
        }
    }
    
}

int main(int argc, const char * argv[]) {
    
    vector<vector<int>> nums = {{-4,-2147483648,6,-7,0}, {-8,6,-8,-6,0}, {2147483647,2,-9,-6,-10}};
    setZeroes(nums);
    
    printTwoDVector(nums);
    
    return 0;
}
