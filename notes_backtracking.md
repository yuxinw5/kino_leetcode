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

