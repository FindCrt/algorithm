//
//  DeapFirstSearch.hpp
//  algorithm
//
//  Created by shiwei on 2019/1/3.
//

#ifndef DeapFirstSearch_hpp
#define DeapFirstSearch_hpp

#include <stdio.h>
#include "TypicalProblems.hpp"

//深度优先搜索的问题：每个问题都有递归和非递归的版本，递归版本简单，有助于构建非递归版本，非递归的实现就是靠的深度搜索方式

#pragma mark - k数之和问题

/*
 “K数之和”问题本质是一个深度搜索的问题。可以更广泛的理解“深度搜索”：现在有一个问题p(n),n是这个问题的规模，你要做选择，有x个选择，每种选择会生成新的问题p(n1),而n1<n，然后不断的做选择，知道问题规模足够小可以直接被解出来。如果你p(n)和它生成的每个p(n1)都连起来，其实就形成了一棵树，而对问题的检索起就是对这棵树的搜索，可以用深度搜索也可以用广度搜索。
 深度搜索的好处是耗费的额外空间小，在不需要求最短路径这种问题时没必要用广度搜索模式。而且用递归的方式解问题，天然搜的是深度搜索的模式。
 再用这种思路来看待“K数之和”问题：在位置i选了第一个数之后，问题就变为了在[i+1,nums.size()-1]范围了选出k-1个数，使得和为target-num[i]。（1）你有多种选择 （2）每种选择会缩化为更小问题 （3）当选择范围==k时，问题就只有一个可能解，直接得出。
 */


/** 两数之和是目标值 */
vector<vector<int>> twoSum(vector<int> &numbers, int target, int start, int end){
    vector<vector<int>> result;
    
    int left = start, right = end;
    while (left < right) {
        
        int curSum = numbers[left]+numbers[right];
        if (curSum < target) {
            left++;
        }else if (curSum > target){
            right--;
        }else{
            //第一个数不等，则组合不同，避免重复；result里的每组只要和前面的一个不同即可，因为left是单调递增的
            if (result.empty() || numbers[left] != result.back().front()) {
                result.push_back({numbers[left], numbers[right]});
            }
            
            left++;
            right--;
        }
    }
    
    return result;
}

void kSumII(vector<int> &numbers, int k ,int target, int start, vector<vector<int>> &result, vector<int> &answer, int idx);
vector<vector<int>> kSumII_dfs(vector<int> &numbers, int k ,int target);

/** K数之和2： 1.源数据不重复 2.限定数量  */
vector<vector<int>> kSumII(vector<int> &numbers, int k ,int target){
    sort(numbers.begin(), numbers.end());
    
    if (k == 1) {
        int find = binaryFind(numbers, target);
        if (numbers[find]>0) {
            return {{target}};
        }else{
            return {{}};
        }
    }else if (k == 2){
        return twoSum(numbers, target, 0, (int)numbers.size()-1);
    }else{
        //递归求解
        //        vector<vector<int>> result;
        //        vector<int> answer(k, 0);
        //        kSumII(numbers, k, target, 0, result, answer, 0);
        //        return result;
        //非递归求解
        return kSumII_dfs(numbers, k, target);
    }
}

// K数之和2：递归算法
void kSumII(vector<int> &numbers, int k ,int target, int start, vector<vector<int>> &result, vector<int> &answer, int idx){
    
    //出口，使用找2数的方法
    if (k == 2) {
        auto result2 = twoSum(numbers, target, start, (int)numbers.size()-1);
        for (auto &res: result2){
            answer[idx] = res[0];
            answer[idx+1] = res[1];
            result.push_back(answer);
        }
        return;
    }
    
    for (int i = start; i<numbers.size(); i++) {
        answer[idx] = numbers[i];
        kSumII(numbers, k-1, target-numbers[i], i+1, result, answer, idx+1);
    }
}

//定义节点模型包含两部分：1. 递归函数的参数里会变化的那些参数 2.分支选择的标记
struct KSumNode {
    int k;
    int target;
    int start;
    int visit;
};

//K数之和2：因为是深度搜索的模型，可以转为非递归处理
vector<vector<int>> kSumII_dfs(vector<int> &numbers, int k ,int target){
    vector<vector<int>> result;
    vector<int> answer(k, 0);
    
    //路径+初始问题转化的根节点
    stack<KSumNode> path;
    path.push({k, target, 0, -1});
    
    int size = (int)numbers.size();
    
    while (!path.empty()) {
        auto &cur = path.top();
        
        //出口/叶节点
        if (cur.k == 2) {
            auto result2 = twoSum(numbers, cur.target, cur.start, size-1);
            for (auto &res : result2){
                answer[k-cur.k] = res[0];
                answer[k-cur.k+1] = res[1];
                result.push_back(answer);
            }
            path.pop();
            continue;
        }
        
        //1. 做出选择
        cur.visit++;
        if (cur.visit==size) { //路径到头
            path.pop();
            continue;
        }
        
        //2. 自身数据更新+解的验证; 因为k==2时有更高效解法，所以解的验证放到了上面。
        answer[k-cur.k] = numbers[cur.visit];
        
        //3. 产生新的问题
        path.push({cur.k-1, cur.target-numbers[cur.visit], cur.visit+1, cur.visit});
    }
    
    return result;
}

int solutionCountOfKSum(vector<int> &numbers, int k, int target){
    sort(numbers.begin(), numbers.end());
    //路径+初始问题转化的根节点
    stack<KSumNode> path;
    path.push({k, target, 0, -1});
    
    int size = (int)numbers.size();
    int soluCount = 0;
    
    while (!path.empty()) {
        auto &cur = path.top();
        
        //出口/叶节点
        if (cur.k == 1) {
            auto range = searchRange(numbers, cur.target, cur.start, size-1);
            if (range[0]>=0) {
                soluCount += range[1]-range[0]+1;
            }
            path.pop();
            continue;
        }
        
        //1. 做出选择
        cur.visit++;
        if (cur.visit==size) { //路径到头
            path.pop();
            continue;
        }
        
        //3. 产生新的问题
        path.push({cur.k-1, cur.target-numbers[cur.visit], cur.visit+1, cur.visit});
    }
    
    return soluCount;
}

//k数之和3：1.有重复，但重复的也算(处理起来跟没重复一样) 2.只求解的数量 3.解只能全部是奇数或全部是偶数
//第3点就是个噱头，把原数组拆成奇数和偶数两个数组，再分别求解就可以了，本质不变
//第2点就是把解的维护删掉，反而是简化了
//比较特别的是第1点，这个在简化到k==2和k==1时是不同的，比如[1,1,1,2,2]，求两数之和为3，不算重复，解只有[1,2],但算重复解的个数是3*2=6，这种情况用原本的左右指针往中间夹的方式是算不全的。然后k==1的时候，还是用二分法，但是是求区间的那一个进化版。
int KSumIII(vector<int> &a, int k, int target) {
    vector<int> oddNum;
    vector<int> evenNum;
    
    for (auto &n : a){
        if (n&1) {
            oddNum.push_back(n);
        }else{
            evenNum.push_back(n);
        }
    }
    
    return solutionCountOfKSum(oddNum, k, target)+solutionCountOfKSum(evenNum, k, target);
}

#pragma mark - 子序列、排列问题

void permuteUnique(vector<int> &nums, int k, bool *mark,vector<vector<int>> &result, vector<int> &answer){
    
    if (k == nums.size()) {
        result.push_back(answer);
        return;
    }
    
    int pre = INT_MIN;
    for (int i = 0; i<nums.size(); i++) {
        if (!mark[i] && nums[i] != pre) {
            mark[i] = true;
            pre = answer[k] = nums[i];
            permuteUnique(nums, k+1, mark, result, answer);
            mark[i] = false;
        }
    }
}

//带有重复元素的排列
vector<vector<int>> permuteUnique(vector<int> &nums) {
    sort(nums.begin(), nums.end());
    bool mark[nums.size()];
    memset(mark, 0, sizeof(mark));
    
    vector<vector<int>> result;
    vector<int> answer(nums.size(), 0);
    
    permuteUnique(nums, 0, mark, result, answer);
    
    return result;
}

struct PermuteUniqueNode {
    int k;  //第几个数的选择
    int idx;   //被选择的数的索引
};

vector<vector<int>> permuteUnique_dfs(vector<int> &nums){
    if (nums.empty()) {
        return {{}};
    }
    
    int size = (int)nums.size();
    sort(nums.begin(), nums.end());
    bool mark[size];
    memset(mark, 0, sizeof(bool)*size);
    
    vector<vector<int>> result;
    vector<int> answer(size, 0);
    
    stack<PermuteUniqueNode> path;
    path.push({0, -1});
    //需要维护的变量：mark,result,answer,path
    while (!path.empty()) {
        
        auto &cur = path.top();
        
        //1.做选择
        int pre = -1;
        if (cur.idx>=0) {  //重置之前的选择相关的数据
            pre = cur.idx;
            mark[pre] = false;
        }
        
        do {
            cur.idx++;
        } while (cur.idx < size &&
                 ((pre >= 0 && nums[cur.idx]==nums[pre]) ||
                  mark[cur.idx]));
        
        if (cur.idx == size) {  //没得选了，回溯
            path.pop();
            continue;
        }
        
        mark[cur.idx] = true;
        answer[cur.k] = nums[cur.idx];
        
        if (cur.k == size-1) {   //得到解了，回溯
            mark[cur.idx] = false;
            result.push_back(answer);
            path.pop();
            continue;
        }
        
        path.push({cur.k+1, -1});
    }
    
    return result;
}

#pragma mark - 生成二叉搜索树

//递归实现
vector<TreeNode *> generateTrees(int start, int end){
    if (start>end) {
        return {nullptr};
    }
    
    vector<TreeNode *> result;
    for (int i = start; i<=end; i++) {
        
        auto lefts = generateTrees(start, i-1);
        auto rights = generateTrees(i+1, end);
        
        for (auto &l : lefts){
            for (auto &r : rights){
                auto node = new TreeNode(i);
                node->left = l;
                node->right = r;
                result.push_back(node);
            }
        }
    }
    
    return result;
}

//生成一棵二叉搜索树，搜索树的特性是左子树节点<=根节点<=右子树节点。
vector<TreeNode *> generateTrees(int n) {
    return generateTrees(1, n);
}

struct GenerateTreeNode {
    int start;
    int end;
    int idx;
    bool searchingLeft;
    vector<TreeNode *> allTrees;
    vector<TreeNode *> lefts;
    vector<TreeNode *> rights;
};

//深度优先搜索模式的非递归
//这一题特别的地方在于解不在路径末端，而是把自身解返回，也就是“从头到尾，再回到头”，最后回来的时候才得到完整解。
//多一个把数据传回的操作，标志就是递归函数是有返回值的
/*这一题还有一个特殊点是分支选择是双层的，既要选根节点，又要遍历左右子树，对某个子问题完整过程是：
 1. 选一个根节点
 2. 处理左子树，
 3. 处理右子树
 4. 合并左、右子树形成获取当前问题的解
 循环1.2.3.4直到没有更多的数可以选。对于某个节点而言，它从栈里拿出来时，可能状态有：4+1+2,3。因为这个过程是循环的，所以这个4是上一次选择情况的4，这样划分的好处是只有两种状态要处理。
 完整的这个流程就是:[4*,1,2][3]...[4,1,2][3][4,1*,2*][3*]。标'*'的表示实际不会执行的，也是需要添加额外条件判断的地方。一个中括号内的操作表示一起处理掉。
 */
vector<TreeNode *> generateTrees_dfs(int n){
    vector<GenerateTreeNode> path;
    path.push_back({1,n,-1,false,{},{},{}});
    
    while (1) {
        auto &cur = path.back();
        
        if (!cur.searchingLeft) {
            //状态4
            if (cur.idx >= 0) { //初始化是idx都设成-1，>=0表示之前有选择，进入合成处理
                for (auto &l : cur.lefts){
                    for (auto &r : cur.rights){
                        auto node = new TreeNode(cur.idx);
                        node->left = l;
                        node->right = r;
                        cur.allTrees.push_back(node);
                    }
                }
                
                cur.idx++;
            }else{
                cur.idx = cur.start;
            }
            
            //状态1
            cur.searchingLeft = true;
        }else{
            cur.searchingLeft = false;
        }
        
        //回溯，把值回传
        if (cur.idx > cur.end) {
            
            if (cur.allTrees.empty()) {
                cur.allTrees.push_back(nullptr);
            }
            if (path.size() == 1) {
                return cur.allTrees;
            }
            
            auto &parent = path[path.size()-2];
            if (parent.searchingLeft) {
                parent.lefts = cur.allTrees;
            }else{
                parent.rights = cur.allTrees;
            }
            
            path.pop_back();
        }else{
            
            //状态2
            if (cur.searchingLeft) {
                path.push_back({cur.start, cur.idx-1, -1, false, {}, {}, {}});
            }else{
                //状态3
                path.push_back({cur.idx+1, cur.end, -1, false, {}, {}, {}});
            }
        }
    }
}


#endif /* DeapFirstSearch_hpp */
