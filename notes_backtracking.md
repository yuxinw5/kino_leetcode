## Backtracking

### 78. Subsets

```python
  def subsets(self, nums: List[int]) -> List[List[int]]:
      def dfs(nums,path):
          res.append(path)
          if not nums:
              return
          for i in range(len(nums)):
              dfs(nums[i+1:],path+[nums[i]])
      res = []
      dfs(nums,[])
      return res
```

### 90. Subsets II

可以用path中sort去重，也可以在最开始的nums上sort达到去重。

```python
  def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
      def dfs(nums, path):
          if sorted(path) not in res:
              res.append(sorted(path))
          if not nums:
              return
          for i in range(len(nums)):
              dfs(nums[i+1:],path+[nums[i]])
      res = []
      dfs(nums, [])
      return res
```

```python
def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
      def dfs(nums, path, res):
          if path not in res:
              res.append(path)
          for i in range(len(nums)):
              # if i > 0 and nums[i] == nums[i-1]:
              #     continue
              dfs(nums[i+1:], path+[nums[i]], res)

      res = []
      nums.sort()
      dfs(nums, [], res)
      return res
```

### 77. Combinations

```python
def combine(self, n: int, k: int) -> List[List[int]]:
      nums = [i+1 for i in range(n)]

      def dfs(nums, path, k):
          if k == 0:
              res.append(path)
          for i in range(len(nums)):
              dfs(nums[i+1:], path + [nums[i]], k-1)
      res = []
      dfs(nums, [], k)
      return res
```

### 39. Combination Sum

```python
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
      def dfs(nums, path, target):
          if sum(path) == target:
              res.append(path)
          if sum(path) > target:
              return
          for i in range(len(nums)):
              dfs(nums[i:], path + [nums[i]], target)
      res = []
      dfs(candidates, [], target)
      return res
```

### 40. Combination Sum II

本题数组candidates的元素是有重复的, 如果把所有组合求出来，再用set或者map去重，这么做很容易超时。所以要在搜索的过程中就去掉重复组合。

组合问题可以抽象为树形结构，那么“使用过”在这个树形结构上是有两个维度的，一个维度是同一树枝上使用过，一个维度是同一树层上使用过。

树层去重的话，需要对数组排序，并且在遍历过程中判断是否已经取用过。if i > 0 and nums[i] == nums[i-1]， 说明前一个树枝，使用了nums[i - 1]，也就是说同一树层使用过nums[i - 1]

```python
  def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
      def dfs(nums, path, target):
          if sum(path) == target:
              res.append(path)
          if sum(path) > target:
              return
          for i in range(len(nums)):
              if i > 0 and nums[i] == nums[i-1]: 
                  continue
              dfs(nums[i+1:], path + [nums[i]], target)
      res = []
      dfs(sorted(candidates), [], target)
      return res
```

### 216. Combination Sum III

```python
  def combinationSum3(self, k: int, n: int) -> List[List[int]]:
      nums = [i+1 for i in range(9)]

      def dfs(nums, path, k, target):
          if sum(path) == target and k == 0:
              res.append(path)
          if sum(path) > target or k < 0:
              return
          for i in range(len(nums)):
              dfs(nums[i+1:], path + [nums[i]], k-1, target)
      res = []
      dfs(nums, [], k, n)
      return res
```

### 377. Combination Sum IV

如果本题要把排列都列出来的话，只能使用回溯算法爆搜。但只是计数问题，所以用dp就够了。而且名字叫组合，其实题目要求是求排列。

个数可以不限使用，说明这是一个完全背包。如果求排列数就是外层for遍历背包，内层for循环遍历物品。

```python
def combinationSum4(self, nums: List[int], target: int) -> int:
      dp = [0] * (target + 1)
      dp[0] = 1

      for j in range(target + 1):
          for i in range(len(nums)):
              if j >= nums[i]:
                  dp[j] += dp[j - nums[i]]
      return dp[-1]
```

### 254. Factor Combinations

Numbers can be regarded as product of its factors. For example,

8 = 2 x 2 x 2 = 2 x 4.
  
Write a function that takes an integer n and return all possible combinations of its factors.

思路：DFS + backtracking。每次搜索从最小的一个因子开始到sqrt(n)查看之间的数是否可以被n整除，如果可以那么有两种选择：

1. n 除以这个因子，并保持这个因子， 然后进行下一层搜索。

2. 不再进行下一层搜索，保存当前结果。

```python
def getFactors(self, n: int) -> List[List[int]]:
    res = []
    self.helper(res, 2, [], n)
    return res

def helper(self, res, start, temp, remain):
    if remain==1 and len(temp)>1:
        res.append(temp)
    else:    
        for i in range(start, remain+1):
            if remain%i == 0:
                self.helper(res, i, temp+[i], remain//i)
```

### 46. Permutations

```python
def permute(self, nums: List[int]) -> List[List[int]]:
    def dfs(nums, path):
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            dfs(nums[:i] + nums[i+1:], path+[nums[i]])
    res = []
    dfs(nums, [])
    return res
```

### 47. Permutations II

```python
def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    def dfs(nums, path):
        if not nums and path not in res:
            res.append(path)
        for i in range(len(nums)):
            dfs(nums[:i] + nums[i+1:], path+[nums[i]])
    res = []
    dfs(nums, [])
    return res
```

### 31. Next Permutation

https://www.nayuki.io/page/next-lexicographical-permutation-algorithm

一个需要死记硬背的算法

```python
def nextPermutation(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    # To find next permutations, we'll start from the end
    i = j = len(nums)-1
    # First we'll find the first non-increasing element starting from the end
    while i > 0 and nums[i-1] >= nums[i]:
        i -= 1
    # After completion of the first loop, there will be two cases
    # 1. Our i becomes zero (This will happen if the given array is sorted decreasingly). In this case, we'll simply reverse the sequence and will return 
    if i == 0:
        nums.reverse()
        return 
    # 2. If it's not zero then we'll find the first number grater then nums[i-1] starting from end
    while nums[j] <= nums[i-1]:
        j -= 1
    # Now out pointer is pointing at two different positions
    # i. first non-assending number from end
    # j. first number greater than nums[i-1]

    # We'll swap these two numbers
    nums[i-1], nums[j] = nums[j], nums[i-1]

    # We'll reverse a sequence strating from i to end
    nums[i:]= nums[len(nums)-1:i-1:-1]
```

### 60. Permutation Sequence

### 291. Word Pattern II

Given a pattern and a string str, find if str follows the same pattern.

Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty substring in str.

Example 1:

Input: pattern = "abab", str = "redblueredblue"

Output: true

Example 2:

Input: pattern = "aabb", str = "xyzabcxzyabc"

Output: false

这道题跟上面一道题290的唯一区别在于，现在的S是连续的。对于连续的S，比较常见的做法就是用backtracking进行暴力搜索。通过backtracking将他分割成所有可能的combination，并且判断双向的对应关系是否符合。具体流程如下：

1. 给定一个pattern所在的位置i，s中所在的位置j
2. 首先判断pattern[i]有没有已经存在的映射，如果有的话，那么现在s[j:j+k]对应的word要和现在pattern[i]对应的word相同，然后进行下一层matching，否则证明已经不符合matching
3. 如果pattern[i]不存在已知映射，那么我们应该枚举pattern[i]映射到以j为index开始的任意的s中的sub string对应，这边就需要backtracking

```python
def wordPatternMatch(self, pattern: str, str: str) -> bool:
    def match(i,j):
        is_match = False
        if i==len(pattern) and j==len(str):
            is_match = True

        elif i<len(pattern) and j<len(str):
            p = pattern[i]
            if p in p2w:
                w = p2w[p]
                if w==str[j:j+len(w)]:
                    is_match = match(i+1,j+len(w))
            else:
                for k in range(j,len(str)):
                    w = str[j:k+1]
                    if w not in w2p:
                        w2p[w],p2w[p] = p,w
                        is_match = match(i+1,k+1)
                        w2p.pop(w)
                        p2w.pop(p)

                    if is_match:
                        break
        return is_match

    w2p = {}
    p2w = {}
    return match(0,0)
```

### 17. Letter Combinations of a Phone Number

下面这样写虽然结果是正确的，但思路完全是混乱的！for i in range(len(nums)) 这一行本质是为了收缩可选择的范围，参考题目77，是为了从左向右取数，取过的数，不在重复取。因此下面这样写就会导致有很多结果略过了前面的数，比如“23”直接从3开始取数。虽然len(path) == len(digits)保证了最后的结果是正确的，但做了很多无用功。

本题最明显的特征是每一个数字代表的是不同集合，也就是求不同集合之间的组合，不存在取还是不取/从哪开始取的问题，每一步都需要取用，所以只能一个digit向后推。因此第二种写法更正确。

```python
def letterCombinations(self, digits: str) -> List[str]:
    dic = {2:['a','b','c'], 3:['d','e','f'], 4:['g','h','i'], 5:['j','k','l'], 6:['m','n','o'], 7:['p','q','r','s'], 8:['t','u','v'], 9:['w','x','y','z']}

    def dfs(nums, path):
        if len(path) == len(digits):
            res.append(path)
        for i in range(len(nums)):
            for c in dic[int(nums[i])]:
                dfs(nums[i+1:], path + c)

    if digits == "":
        return []
    res = []
    dfs(list(digits), "")
    return res
```

```python
def letterCombinations(self, digits: str) -> List[str]:
    dic = {2:['a','b','c'], 3:['d','e','f'], 4:['g','h','i'], 5:['j','k','l'], 6:['m','n','o'], 7:['p','q','r','s'], 8:['t','u','v'], 9:['w','x','y','z']}

    def dfs(nums, path):
        if not nums:
            res.append(path)
            return
        for c in dic[int(nums[0])]:
            dfs(nums[1:], path + c)

    if digits == "":
        return []
    res = []
    dfs(list(digits), "")
    return res
```

### 320. Generalized Abbreviation

Write a function to generate the generalized abbreviations of a word.

Input: "word"

Output: ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]

https://www.cnblogs.com/grandyang/p/5261569.html

这个题没看明白

### 282. Expression Add Operators

Given a string that contains only digits 0-9 and a target value, return all possibilities to add binary operators (not unary) +, -, or * between the digits so they evaluate to the target value.

Example 1:

Input: num = "123", target = 6
Output: ["1+2+3", "1*2*3"] 

Example 2:

Input: num = "232", target = 8
Output: ["2*3+2", "2+3*2"]

https://github.com/zhuifengshen/awesome-leetcode/blob/master/docs/Leetcode_Solutions/Python/0282._Expression_Add_Operators.md

我们要注意的是一个式子不能以+,-,* 开头，并且不能出现以‘0’开头的数字。时间复杂度: O(4^n)- 空间复杂度: O(N)

```python
def addOperators(self, num, target):
    def addOperators(self, num: str, target: int) -> List[str]:
        res = []
        for i in range(len(num)):
            if i!=0 and num[0]=='0':
                break
            self.helper(num[i+1:], target, num[:i+1], 0, int(num[:i+1]), res)
        return res

    def helper(self, num, target, path, temp, last, res):
        if not num:
            if temp+last==target:
                res.append(path)
            return
        
        for i in range(len(num)):
            if i!=0 and num[0]=='0':
                break
            number = num[:i+1]
            self.helper(num[i+1:], target, path+"*"+number, temp, last*int(number), res)
            self.helper(num[i+1:], target, path+"+"+number, temp+last, int(number), res)
            self.helper(num[i+1:], target, path+"-"+number, temp+last, -int(number), res)
```

### 140. Word Break II

传统backtrack做法，可以直接用i来表示词的可能结尾index，并不需要另一个loop来遍历。

```python
def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:                
    def dfs(s, path):
        if not s:
            res.append(path)
        for i in range(1,len(s)+1):
            if s[:i] in wordDict:
                if path == "":
                    dfs(s[i:], path + s[:i])
                else:
                    dfs(s[i:], path +" "+ s[:i])
    res = []
    dfs(s,"")
    return res
```

### 351. Android Unlock Patterns

Given an Android 3x3 key lock screen and two integers m and n, where 1 ≤ m ≤ n ≤ 9, count the total number of unlock patterns of the Android lock screen, which consist of minimum of m keys and maximum n keys.

Rules for a valid pattern:

1. Each pattern must connect at least m keys and at most n keys.
2. All the keys must be distinct.
3. If the line connecting two consecutive keys in the pattern passes through any other keys, the other keys must have previously selected in the pattern. No jumps through non selected key is allowed.
4. The order of keys used matters.

https://www.cnblogs.com/grandyang/p/5541012.html

https://github.com/xiaoningning/LeetCode-python/blob/master/351%20Android%20Unlock%20Patterns.py

这道题乍一看题目这么长以为是一个设计题，其实不是，这道题还是比较有意思的，起码跟实际结合的比较紧密。这道题说的是安卓机子的解锁方法，有9个数字键，如果密码的长度范围在 [m, n] 之间，问所有的解锁模式共有多少种，注意题目中给出的一些非法的滑动模式。那么先来看一下哪些是非法的，首先1不能直接到3，必须经过2，同理的有4到6，7到9，1到7，2到8，3到9，还有就是对角线必须经过5，例如1到9，3到7等。建立一个二维数组 jumps，用来记录两个数字键之间是否有中间键，然后再用一个一位数组 visited 来记录某个键是否被访问过，然后用递归来解，先对1调用递归函数，在递归函数中，遍历1到9每个数字 next，然后找他们之间是否有 jump 数字，如果 next 没被访问过，并且 jump 为0，或者 jump 被访问过，对 next 调用递归函数。数字1的模式个数算出来后，由于 1,3,7,9 是对称的，所以乘4即可，然后再对数字2调用递归函数，2,4,6,9 也是对称的，再乘4，最后单独对5调用一次

```python
class Solution(object):
    def __init__(self):
        self.skip = [[None for _ in xrange(10)] for _ in xrange(10)]
        self.skip[1][3], self.skip[3][1] = 2, 2
        self.skip[1][7], self.skip[7][1] = 4, 4
        self.skip[3][9], self.skip[9][3] = 6, 6
        self.skip[7][9], self.skip[9][7] = 8, 8
        self.skip[4][6], self.skip[6][4] = 5, 5
        self.skip[2][8], self.skip[8][2] = 5, 5
        self.skip[1][9], self.skip[9][1] = 5, 5
        self.skip[3][7], self.skip[7][3] = 5, 5

    def numberOfPatterns(self, m, n):
        visited = [False for _ in xrange(10)]
        return sum(
            self.dfs(1, visited, remain) * 4 +
            self.dfs(2, visited, remain) * 4 +
            self.dfs(5, visited, remain)
            for remain in xrange(m, n+1)
        )

    def dfs(self, cur, visited, remain):
        if remain == 1:
            return 1

        visited[cur] = True
        ret = 0
        for nxt in xrange(1, 10):
            if (
                not visited[nxt] and (
                    self.skip[cur][nxt] is None or
                    visited[self.skip[cur][nxt]]
                )
            ):
                ret += self.dfs(nxt, visited, remain - 1)

        visited[cur] = False
        return ret
```
