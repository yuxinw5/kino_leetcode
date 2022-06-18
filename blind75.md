## Blind 75 LeetCode Questions

https://leetcode.com/discuss/general-discussion/460599/blind-75-leetcode-questions

## Array

### 1. Two Sum

We use dictionary to store the index. It takes O(n) because dictionary and set in python works on Hashing which is an approach to return a unique value for every value passed in Hashing function. Thus fetching elements from Dictionary and Set in O(1) operation.

```python
  def twoSum(self, nums: List[int], target: int) -> List[int]:
      d = {}
      for i, n in enumerate(nums):
          if target - n not in d:
              d[n] = i
          else:
              return [d[target - n], i]
```

### 121. Best Time to Buy and Sell Stock

#### Solution: Brutal Force

O(n^2), 用两个loop找最优间距

#### Solution: Advanced Brutal Force

O(n),记录遇见过的最低买入价格，跟当前价格做差价，记录最大收益

```python
def maxProfit(prices):
    max_profit, min_price = 0, float('inf')
    for price in prices:
        min_price = min(min_price, price)
        profit = price - min_price
        max_profit = max(max_profit, profit)
    return max_profit
```

#### Solution: DP

如果第i天持有股票即dp[i][0]， 那么可以由两个状态推出来，选最大的

1. 第i-1天就持有股票，那么就保持现状，所得现金就是昨天持有股票的所得现金 即：dp[i - 1][0]
2. 第i天买入股票，所得现金就是买入今天的股票后所得现金即：-prices[i]

如果第i天不持有股票即dp[i][1]， 也可以由两个状态推出来，选最大的

1. 第i-1天就不持有股票，那么就保持现状，所得现金就是昨天不持有股票的所得现金 即：dp[i - 1][1]
2. 第i天卖出股票，所得现金就是按照今天股票佳价格卖出后所得现金即：prices[i] + dp[i - 1][0]

```python
def maxProfit(self, prices: List[int]) -> int:
    dp = [[0] * 2 for _ in range(len(prices))]
    dp[0][0] = -prices[0] # buy
    dp[0][1] = 0 # sell

    for i in range(1,len(prices)):
        dp[i][0] = max(dp[i-1][0], -prices[i])
        dp[i][1] = max(dp[i-1][1], prices[i] + dp[i-1][0])
    return dp[-1][1]
```

### 217. Contains Duplicate

Take advantage of python

```python
def containsDuplicate(self, nums):
    return len(nums) != len(set(nums))
```

### 238. Product of Array Except Self

O(n) Use prefix array and suffix array. Prefix/Suffix array represent the product of all elements before/after the current element. The final result is the product of the two arrays.

```python
def productExceptSelf(self, nums: List[int]) -> List[int]:
    pre = [1] * len(nums)
    suf = [1] * len(nums)

    for i in range(1, len(nums)):
        pre[i] = pre[i-1] * nums[i-1]
    for i in range(len(nums)-2, -1, -1):
        suf[i] = suf[i+1] * nums[i+1]

    res = []
    for i in range(len(nums)):
        res.append(pre[i]*suf[i])

    return res
```

### 53. Maximum Subarray

#### Solution: DP

At each index, keep track of the maximum sum using DP table , till that point:

1. Save the maximum between [cur_value, max_so_far+cur_value]
2. Finally, return the maximum out of the table

```python
def maxSubArray(self, nums):
		dp = [0] * len(nums)
    for i,num in enumerate(nums):            
        dp[i] = max(dp[i-1] + num, num)
    return max(dp)
```

#### Solution: Kadane's Algorithm V1

If the sum of a subarray is positive, it has possible to make the next value bigger, so we keep do it until it turn to negative.

If the sum is negative, it has no use to the next element, so we break.

```python
def maxSubArray(self, nums):
    cur_max, max_till_now = 0, -inf
    for c in nums:
        cur_max = max(c, cur_max + c)
        max_till_now = max(max_till_now, cur_max)
    return max_till_now
```

#### Solution: Kadane's Algorithm V2

```python
def maxSubArray(self, nums: List[int]) -> int:
    for i in range(1, len(nums)):
        if nums[i-1] > 0:
            nums[i] += nums[i-1]
    return max(nums)
```

### 152. Maximum Product Subarray

#### Solution: DP

The trick is

1. Unlike maximum sum, for max product the current negetive minimum value could be max in future when multiplied by another -ve value --> so we need to keep track of minimum_value too
2. We need to keep track of maximum value --> obviously
3. We need to keep track of over_all maximum value

We again use dpMax and dpMin arrays where dpMax[i] denotes maximum subarray product ending at i and dpMin[i] denotes minimum subarray product ending at i.

```python
def maxProduct(self, nums: List[int]) -> int:
    dpmin = [0] * len(nums)
    dpmax = [0] * len(nums)
    dpmin[0] = dpmax[0] = nums[0]

    for i in range(1, len(nums)):
        num = nums[i]
        if num >= 0:
            dpmin[i] = min(dpmin[i-1] * num, num)
            dpmax[i] = max(dpmax[i-1] * num, num)
        else:
            dpmin[i] = min(dpmax[i-1] * num, num)
            dpmax[i] = max(dpmin[i-1] * num, num)

    return max(dpmax)
```

#### Solution: DP optimized

```python
def maxProduct(self, nums):
    max_prod, min_prod, ans = nums[0], nums[0], nums[0]
    for i in range(1, len(nums)):
        x = max(nums[i], max_prod*nums[i], min_prod*nums[i])
        y = min(nums[i], max_prod*nums[i], min_prod*nums[i])            
        max_prod, min_prod = x, y
        ans = max(max_prod, ans)
    return ans
```

One important thing to note here is that we've to use old max_prod and old min_prod in every iteration to compute their updated values and not updated max_prod or updated min_prod. Therefore, we need the x and y variables.

### 153. Find Minimum in Rotated Sorted Array

#### Solution: Binary Search in general case

Find the index of a local peak value in O(logn)

```python
def findPeak(arr):
    n = len(arr) 
    l = 0
    r = n-1
     
    while(l <= r):
        mid = (l + r) >> 1
 
        if((mid == 0 or arr[mid - 1] <= arr[mid]) and (mid == n - 1 or arr[mid + 1] <= arr[mid])):
            break
 
        if (mid > 0 and arr[mid - 1] > arr[mid]):
            r = mid - 1
        else:
            l = mid + 1
 
    return mid
```

#### Solution: Binary Search

1. loop is left < right, which means inside the loop, left always < right
2. since we use round up for mid, and left < right from (1), right would never be the same as mid
3. Therefore, we compare mid with right, since they will never be the same from (2)
4. if nums[mid] < nums[right], we will know the minimum should be in the left part, so we are moving right.
5. if nums[mid] > nums[right], minimum should be in the right part, so we are moving left. Since nums[mid] > nums[right],mid can't be the minimum, we can safely move left to mid + 1, which also assure the interval is shrinking

Why we compare nums[mid] and nums[right] instead of nums[left]?

It's best to think about the binary search conditions (for any problem that uses it) in terms of which parts of the array get disqualified for the next iteration. In this case here, we want to find the minimum value and we know the array is "rotated sorted." So, whenever the condition nums[mid] > nums[left] holds, that of course means the value nums[left] is less than nums[mid]. However, it is not a guarantee that there are no smaller values in the right side because the array is rotated.

```python
def findMin(self, nums: List[int]) -> int:
    l = 0
    r = len(nums)-1
    while l < r:
        mid = (l+r)//2
        if nums[mid] > nums[r]:
            l = mid+1
        else:
            r = mid
    return nums[l]
```

### 33. Search in Rotated Sorted Array

#### Solution: Binary Search

If nums[low] <= nums[mid], the left side is strictly increasing. Otherwise, the right side is strictly increasing. Then we determine whether the target is in the ascending order sise.

```python
def search(self, nums: List[int], target: int) -> int:
    if not nums:
        return -1

    low, high = 0, len(nums) - 1

    while low <= high:
        mid = (low + high) // 2
        if target == nums[mid]:
            return mid

        if nums[low] <= nums[mid]:
            if nums[low] <= target <= nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        else:
            if nums[mid] <= target <= nums[high]:
                low = mid + 1
            else:
                high = mid - 1
    return -1
```

### 15. 3Sum

#### Solution: Two Pointers

传统双指针，通过大小控制指针运动方向。注意去重剪枝的操作，可以加快运行速度，但对正确与否没有影响。

```python
def threeSum(self, nums: List[int]) -> List[List[int]]:
	res = set()
	nums.sort()
	for i in range(len(nums)-2):
	    if i > 0 and nums[i] == nums[i-1]: #去重剪枝
		continue
	    l, r = i+1, len(nums)-1
	    while l < r:
		s = nums[i] + nums[l] + nums[r]
		if s < 0:
		    l +=1 
		elif s > 0:
		    r -= 1
		else:
		    res.add((nums[i], nums[l], nums[r]))
		    l += 1
		    r -= 1
	return res
```

### 11. Container With Most Water

#### Solution: Greedy Two Pointers

First, we check the widest possible container starting from the first line to the last one. Next, we ask ourselves, how is it possible to form an even bigger container? Every time we narrow the container, the width becomes smaller so the only way to get a bigger area is to find higher lines. So we just greedily shrink the container on the side that has a shorter line. O(n)

```python
def maxArea(self, height: List[int]) -> int:
	l = 0
	r = len(height)-1
	res = (r - l) * min(height[l], height[r])
	while l < r:
	    if height[l] < height[r]:
		l += 1
	    else:
		r -= 1
	    res = max(res, (r - l) * min(height[l], height[r]))

	return res
```

## Binary

### 371. Sum of Two Integers

The first step is to manually bound the length of sum and carry by setting up a mask 0xFFFFFFFF. & this mask with an (very long) integer will only keep the last 32 bits. Then, at each step of the loop, we & sum and carry with this mask, and eventually carry will be wiped out once it goes beyond 32 bits.

Although we successfully get out of the while loop, sadly a consequence is that the value of a also has only 32 bits. If it is a non-negative value then we are fine, but we will lose information if it is negative in the normal sense. So we flip all the bits, then plus one.

```python
def getSum(self, a: int, b: int) -> int:
	mask = 0xffffffff
	a = a & mask
	while b:
	    uncarry_sum = (a^b) & mask
	    carry = ((a&b)<<1) & mask
	    a = uncarry_sum
	    b = carry
	if (a>>31) & 1: # If a is negative in 32 bits sense
	    return ~(a^mask)
	return a
```

### 191. Number of 1 Bits

If we have number n, then n&(n-1) will remove the rightmost in binary representation of n. For example if n = 10110100, then n & (n-1) = 10110100 & 10110011 = 10110000, where & means bitwize operation and. What we need to do now, just repeat this operation until we have n = 0 and count number of steps.

```python
def hammingWeight(self, n):
	ans = 0
	while n:
		n &= (n-1)
		ans += 1
	return ans
```

### 338. Counting Bits

```python
def countBits(self, n: int) -> List[int]:
	counter = [0]
	for i in range(1, n+1):
		counter.append(counter[i >> 1] + i % 2)
	return counter
```

### 268. Missing Number

```python
def missingNumber(self, nums: List[int]) -> int:
	n = len(nums)
	return int(n * (n+1) / 2 - sum(nums)) # get sum of 1 to n
```

### 190. Reverse Bits

Create number out, process original number bit by bit from end and add this bit to the end of our out number.

1. out = (out << 1)^(n & 1) adds last bit of n to out, ^ represents XOR
2. n >>= 1 removes last bit from n.

```python
def reverseBits(self, n: int) -> int:
	out = 0
	for i in range(32):
		out = (out << 1)^(n & 1)
		n >>= 1
	return out
```

## Dynamic Programming

### 70. Climbing Stairs

```python
def climbStairs(self, n: int) -> int:
	if n == 1 or n == 2:
		return n
	dp = [0] * (n+1) # considering zero steps we need n+1 places
	dp[1] = 1
	dp[2] = 2
	for i in range(3, n+1):
		dp[i] = dp[i-1] + dp[i-2]

	return dp[-1]
```

### 322. Coin Change

完全背包： https://programmercarl.com/0322.%E9%9B%B6%E9%92%B1%E5%85%91%E6%8D%A2.html#_322-%E9%9B%B6%E9%92%B1%E5%85%91%E6%8D%A2

```python
def coinChange(self, coins: List[int], amount: int) -> int:
	dp = [amount + 1] * (amount + 1)
	dp[0] = 0

	for c in coins:
		for j in range(c, amount+1):
			dp[j] = min(dp[j], dp[j-c] + 1)

	return dp[amount] if dp[amount] < amount + 1 else -1
```

### 300. Longest Increasing Subsequence

dp[i]表示i之前包括i的以nums[i]结尾最长上升子序列的长度。 注意递推公式里面，不是要dp[i] 与 dp[j] + 1进行比较，而是我们要取dp[j] + 1的最大值，dp[i]起的是记录作用。

```python
def lengthOfLIS(self, nums: List[int]) -> int:
	if len(nums) <= 1:
		return len(nums)
	dp = [1] * len(nums)
	for i in range(1, len(nums)):
		for j in range(0, i):
			if nums[i] > nums[j]:
				dp[i] = max(dp[i], dp[j] + 1)
	return max(dp)
```

### 1143. Longest Common Subsequence

根据递推公式，注意遍历顺序，要从前向后，从上到下来遍历。

```python
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
	len1, len2 = len(text1)+1, len(text2)+1
	dp = [[0 for _ in range(len1)] for _ in range(len2)]
	for i in range(1, len2):
		for j in range(1, len1):
			if text1[j-1] == text2[i-1]:
				dp[i][j] = dp[i-1][j-1] + 1 
			else:
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
	return dp[-1][-1]
```

### 139. Word Break

d[i] is True if there is a word in the dictionary that ends at ith index of s AND d is also True at the beginning of the word

IF statement meaning: does the current word end at this index AND (did a word end before the start of current word OR does this current word start from the beginning of the string)

```python
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
	dp = [False] * len(s)    
	for i in range(len(s)):
		for w in wordDict:
			if w == s[i-len(w)+1:i+1] and (dp[i-len(w)] or i-len(w) == -1):
				dp[i] = True
	return dp[-1]
```

### 377. Combination Sum IV

完全背包 排列总和: https://programmercarl.com/0377.%E7%BB%84%E5%90%88%E6%80%BB%E5%92%8C%E2%85%A3.html#_377-%E7%BB%84%E5%90%88%E6%80%BB%E5%92%8C-iv

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

### 198. House Robber

```python
def rob(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[1]

        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            dp[i] = max(nums[i] + dp[i-2], dp[i-1])
        return dp[-1]
```

### 213. House Robber II

Now, what we have here is circular pattern. Imagine, that we have 10 houses: a0, a1, a2, a3, ... a9: Then we have two possible options:

1. Rob house a0, then we can not rob a1 or a9 and we have a2, a3, ..., a8 range to rob
2. Do not rob house a0, then we have a1, a2, ... a9 range to rob.

```python
def rob(self, nums: List[int]) -> int:

	def rob_simple(nums):
		if len(nums) == 0:
			return 0
		if len(nums) == 1:
			return nums[0]
		dp = [0] * len(nums)
		dp[0] = nums[0]
		dp[1] = max(nums[0], nums[1])

		for i in range(2, len(nums)):
			dp[i] = max(nums[i] + dp[i-2], dp[i-1])
		return dp[-1]

	return max(nums[0] + rob_simple(nums[2:-1]), rob_simple(nums[1:]))
```

### 91. Decode Ways

dp[i] = number of ways to decode the string ending at index i

注意out of range的处理

```python
def numDecodings(self, s: str) -> int:
	dp = [0] * (len(s))
	dp[0] = 0 if s[0] == "0" else 1

	for i in range(1, len(s)): 
		if 0 < int(s[i]) <= 9:
			dp[i] += dp[i - 1]

		if 10 <= int(s[i-1:i+1]) <= 26:
			if i > 1:
				dp[i] += dp[i - 2]
			else:
				dp[i] += 1

	return dp[-1]
```

### 62. Unique Paths

```python
def uniquePaths(self, m: int, n: int) -> int:
	dp = [[1 for _ in range(n)] for _ in range(m)]

	for i in range(1,m):
		for j in range(1,n):
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
	return dp[-1][-1]
```

### 55. Jump Game

#### Solution: Greedy

https://programmercarl.com/0055.%E8%B7%B3%E8%B7%83%E6%B8%B8%E6%88%8F.html#%E6%80%9D%E8%B7%AF

```python
def canJump(self, nums: List[int]) -> bool:
	if len(nums) == 1: return True
	i = cover = 0
	while i <= cover:
		cover = max(cover, i + nums[i])
		if cover >= len(nums)-1:
			return True
		i += 1

	return False
```

#### Solution: DP TLE

dp[i] = whether it can reach stair i from the first stair

```python
def canJump(self, nums: List[int]) -> bool:
        dp = [False] * len(nums)
        dp[0] = True
        
        for i in range(len(nums)):
            for j in range(i):
                if dp[j] and (j + nums[j] >= i):
                    dp[i] = True
        return dp[-1]
```

#### Solution: DP

DP definition: The farthest index we can reach given allowed steps from 0 to i

DP decision & relationship: It's either the dp[i - 1] or i + nums[i] whichever one is larger

DP condition:

1. If at any moment, dp[i-1] < i, that means there is no way it can reach i, return False immediately.
2. If at any moment, dp[i] >= last index, that means it can already reach the end of the array given the steps allowed from 0 to i, return True immediately.

```python
def canJump(self, nums: List[int]) -> bool:
	dp = [0] * len(nums)
	dp[0] = nums[0]

	for i in range(1, len(nums) - 1):
		if dp[i - 1] < i:
			return False

		dp[i] = max(i + nums[i], dp[i - 1])

		if dp[i] >= len(nums) - 1:
			return True

	return dp[len(nums) - 2] >= len(nums) - 1
```

## Graph

### 133. Clone Graph

We use ditionary as a visited set and also use it to record the mapping from the original graph to the new graph

```python
def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return None
        q = deque([node])
        mapping ={node : Node(node.val,[])}
        while q:
            n = q.popleft()
            for i in n.neighbors:
                if i not in mapping:
                    mapping[i] = Node(i.val,[])
                    q.append(i)
                mapping[n].neighbors.append(mapping[i])
        return mapping[node]
```

### 207. Course Schedule

Cycle Detection: NOT_CHECKED, CHECKING, COMPLETED = 0, -1, 1

https://leetcode.com/problems/course-schedule/discuss/658379/Python-by-DFS-and-cycle-detection-w-Graph

```python
def canFinish(self, n: int, prerequisites: List[List[int]]) -> bool:
	adjList = defaultdict(list)
	for i in prerequisites:
		adjList[i[0]].append(i[1])
	state = [0] * n

	def hasCycle(course):
		if state[course] == 1:
			return False
		if state[course] == -1:
			return True
		state[course] = -1
		for i in adjList[course]:
			if hasCycle(i):
				return True
		state[course] = 1
		return False

	for i in range(n):
		if hasCycle(i):
			return False
	return True
```

### 417. Pacific Atlantic Water Flow

#### Solution: DFS

DFS template

```python
def prob(matrix):
    # 1. Check for an empty graph.
    if not matrix:
        return []

    # 2. Initialize
    m, n = len(matrix), len(matrix[0])
    visited = set()

    def dfs(i, j):
        # a. Check if visited
        if (i, j) in visited:
            return
		# b. Else add to visted
        visited.add((i, j))

        # c. Traverse neighbors.
	for (x,y) in (i+1,j), (i-1,j), (i,j+1), (i,j-1):
            if 0 <= x < m and 0 <= y < n:
                # d. Add in your question-specific checks.
                traverse(x, y)

    # 3. For each point, traverse it.
    for i in range(m):
        for j in range(n):
            dfs(i, j)
```

For this question, we use dfs twice and find the union of the two visited sets(whether it can be reached from a ocean).

```python
def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
	p_visited = set()
	a_visited = set()
	m, n = len(heights), len(heights[0])

	def traverse(i, j, visited):
	    if (i, j) in visited:
		return
	    visited.add((i, j))

	    for (x,y) in (i+1,j), (i-1,j), (i,j+1), (i,j-1):
		if 0 <= x < m and 0 <= y < n:
		    if heights[x][y] >= heights[i][j]:
			traverse(x, y, visited)

	for row in range(m):
	    traverse(row, 0, p_visited)
	    traverse(row, n - 1, a_visited)

	for col in range(n):
	    traverse(0, col, p_visited)
	    traverse(m - 1, col, a_visited)

	return list(p_visited & a_visited)
```

### 200. Number of Islands

Another way to keep track of the visited nodes is to edit directly on the matrix.

```python
def numIslands(self, grid: List[List[str]]) -> int:
	m = len(grid)
	n = len(grid[0])

	def dfs(i,j):
		grid[i][j] = "2"
		for x,y in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
			if 0<=x<m and 0<=y<n and grid[x][y] == "1":
				dfs(x,y)

	count = 0
	for i in range(m):
		for j in range(n):
			if grid[i][j] == "1":
				dfs(i,j)
				count += 1
	return count
```

### 128. Longest Consecutive Sequence

Expand the sequence from one element and delete visited number along the way. The input num is converted into a set, so each "i in num" is now O(1).

```python
def longestConsecutive(self, nums: List[int]) -> int:
	nums = set(nums)
	maxlen = 0
	while nums:
	    first = last = nums.pop()
	    while first - 1 in nums:
		first -= 1
		nums.remove(first)
	    while last + 1 in nums:
		last += 1
		nums.remove(last)
	    maxlen = max(maxlen, last - first + 1)
	return maxlen
```

### 269. Alien Dictionary

There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. You receive a list of words from the dictionary, wherewords are sorted lexicographically by the rules of this new language. Derive the order of letters in this language.

For example, given the following words in dictionary,

[
  "wrt",
  "wrf",
  "er",
  "ett",
  "rftt"
]

The correct order is: "wertf".

没太学会：https://www.youtube.com/watch?v=6kTZYvNNyps

### 261. Graph Valid Tree

Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

Input: n = 5, and edges = [[0,1], [0,2], [0,3], [1,4]]

Output: true

#### Solution: DFS

https://www.youtube.com/watch?v=bXsUuownnoQ&list=PLot-Xpze53ldVwtstag2TL4HQhAnC8ATf&index=32

Tree properties:
1. connected: all nodes are visited n == len(visited)
2. has |V|-1 edges: len(edges) == n - 1
3. has no cycles: visited set

```python
def validTree(self, n: int, edges: List[List[int]]) -> bool:
	if not n: return True
	adj = collections.defaultdict(list)
	for n1,n2 in edges:
		adj[n1].append(n2)
		adj[n2].append(n1)
	visited = set()
	
	def dfs(i, prev):
		if i in visited:
			return False
		visited.add(i)
		for j in adj[i]:
			if j == prev:
				continue
			if not dfs(j, i):
				return False
		return True
		
	return dfs(0,-1) and n == len(visited)
```

#### Solution: BFS

https://zhenyu0519.github.io/2020/03/28/lc261/#261-graph-valid-tree-python

For this question,

1. As a tree, the the number of edges must equal to the number of nodes - 1.
2. If the given input can build up one valid tree, then there should be only one complete graph which include all nodes.

If the number of edges is the number of nodes - 1, we use BFS start from a certain node and find all related to this node and append them to the visited set. If the final size of the visited set is not equal to the number of nodes. That means there are more than 1 graph can be built by given input. Which is False

```python
def validTree(self, n: int, edges: List[List[int]]) -> bool:
	if len(edges)!=n-1:return False
	dist = collections.defaultdict(list)
	for n1,n2 in edges:
		dist[n1].append(n2)
		dist[n2].append(n1)
	visited=set()
	queue=collections.deque([0])
	while queue:
		node = queue.popleft()
		visited.add(node)
		for related in dist[node]:
			if related not in visited:
				visited.add(related)
				queue.append(related)
	return len(visited)==n
```

### 323. Number of Connected Components in an Undirected Graph

Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to find the number of connected components in an undirected graph.

https://zhenyu0519.github.io/2020/03/27/lc323/#sample-io

```python
def countComponents(self, n: int, edges: List[List[int]]) -> int:
	dist = collections.defaultdict(list)
	for source, target in edges:
		dist[source].append(target)
		dist[target].append(source)
	count = 0
	visited=set()
	queue = collections.deque()
	for x in range(n):
		if x in visited:
			continue
		queue.append(x)
		while queue:
			source=queue.popleft()
			if source in visited:
				continue
			visited.add(source)
			for target in dist[source]:
				queue.append(target)
		count+=1
	return count
```

## Interval

### 57. Insert Interval

Use two variables s and e to remember the current interval that is being checked. 注意检查interval位置的三种情况的分类讨论。

```python
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
	s, e = newInterval[0], newInterval[1]
	left, right = [], []

	for i in intervals:
		if i[1] < s:
			left += i,
		elif i[0] > e:
			right += i,
		else:
			s = min(s, i[0])
			e = max(e, i[1])

	return left + [[s,e]] + right
```

### 56. Merge Intervals

Sort the intervals by the start value. Therefore we only need to consider two cases: 

1. no overlap -> simply append the current interval
2. has overlaps -> merge and append to the end of the result

```python
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
	intervals.sort(key=lambda x: x[0])
	res = []
	for i in intervals:
		if not res or res[-1][1] < i[0]:
			res.append(i)
		else:
			last = res.pop()
			s = min(last[0],i[0])
			e = max(last[1],i[1])
			res.append([s,e])
	return res
```

### 252. Meeting Rooms

Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],…] (si < ei), determine if a person could attend all meetings.

Input: [[0,30],[5,10],[15,20]]

Output: false

https://zhenyu0519.github.io/2020/07/12/lc252/#252-meeting-rooms-python

Sort the list by start time and iterate the sorted list. If the current start time is less than previous end time, then there is conflict and you can not attend all meeting.

```python
def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        new_intervals = sorted(intervals, key=lambda x: x[0])
        for i in range(1,len(new_intervals)):
            if new_intervals[i-1][1] > new_intervals[i][0]:
				return False
        return True
```

### 253. Meeting Rooms II

Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],…] (si < ei), find the minimum number of conference rooms required.

Input: [[0, 30],[5, 10],[15, 20]]
Output: 2

https://zhenyu0519.github.io/2020/07/13/lc253/#253-meeting-rooms-ii-python

#### Solution: Heap

When you look at first meeting starting time, you will wonder if there is a meeting ends (that means a meeting room is released), if yes, you will just take that one; otherwise, you will find a new room. 

生成优先队列, 代表已开的房间, 先将时间段按照开始时间排序, 遍历时间段, 如果没有多余的房间, 则将这个会议的加到队列里, 如果有多余的房间且目前会议的起始之间在上个会议的终止时间之后, 更新队列.

```python
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
	size = len(intervals)
	heap = []
	for interval in sorted(intervals):
		if heap and interval[0] >= heap[0]:
			# heappushpop(a, x) pushes x onto a before popping the smallest value
			heapq.heappushpop(heap,interval[1])
		else:
			heapq.heappush(heap,interval[1])
	return len(heap)
```

## Linked List

### 206. Reverse Linked List

搞清楚结构就好了

#### Solution: Iterative

```python
def reverseList(self, head):
    prev = None
    while head:
        curr = head
        head = head.next
        curr.next = prev
        prev = curr
    return prev
```

#### Solution: Recursive

```python
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
	if not head or not head.next:
		return head
	node = self.reverseList(head.next)
	head.next.next = head
	head.next = None
	return node
```

### 141. Linked List Cycle

#### Solution: Mark visited

不需要建visited set，在原有value基础上mark就好

```python
def hasCycle(self, head: Optional[ListNode]) -> bool:
	while head:
		if head.val == None:
			return True
		head.val = None
		head = head.next
	return False
```

#### Solution: Two Pointers Loop Detection

If there is a cycle, fast will catch slow after some loops.

```python
def hasCycle(self, head):
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if slow == fast:
            return True
	return False
```

### 21. Merge Two Sorted Lists

Use dummy node!

```python
def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
	dummy = ListNode()
	head = dummy
	while list1 or list2:
		if list1 and not list2:
			dummy.next = ListNode(list1.val)
			list1 = list1.next
		elif list2 and not list1:
			dummy.next = ListNode(list2.val)
			list2 = list2.next
		else:
			if list1.val < list2.val:
				dummy.next = ListNode(list1.val)
				list1 = list1.next
			else:
				dummy.next = ListNode(list2.val)
				list2 = list2.next
		dummy = dummy.next
	return head.next
```

Simplified version below.

```python
def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
	dummy = p = ListNode()
	while list1 and list2:
		if list1.val < list2.val:
			p.next = list1
			list1 = list1.next
		else:
			p.next = list2
			list2 = list2.next
		p = p.next
	p.next = list1 or list2
	return dummy.next
```

### 23. Merge k Sorted Lists

#### Solution: Devide and Conquer

Instead of merge 2 lists in sequence, we break it into small cases and use recurtion.

```python
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]: 
	def mergeTwoLists(list1, list2):
		dummy = p = ListNode()
		while list1 and list2:
			if list1.val < list2.val:
				p.next = list1
				list1 = list1.next
			else:
				p.next = list2
				list2 = list2.next
			p = p.next
		p.next = list1 or list2
		return dummy.next

	if not lists:
		return None
	if len(lists) == 1:
		return lists[0]
	mid = len(lists) // 2
	l, r = self.mergeKLists(lists[:mid]), self.mergeKLists(lists[mid:])
	return mergeTwoLists(l, r)
```

#### Solution: Heap

相当于同时比较所有lists当前位置的node。

Why we need idx value stored in the heap? If there are duplicate values in the list, we will get "TypeError: '<' not supported between instances of 'ListNode' and 'ListNode'".

```python
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]: 
	q = []
	for idx, head in enumerate(lists):
		if head is not None:
			q.append((head.val, idx, head))
	heapq.heapify(q)
	dummy = ListNode()
	last = dummy
	while q:
		val, idx, node = heapq.heappop(q)
		last.next, last = node, node
		if node.next is not None:
			heapq.heappush(q, (node.next.val, idx, node.next))
	return dummy.next
```

### 19. Remove Nth Node From End of List

https://programmercarl.com/0019.%E5%88%A0%E9%99%A4%E9%93%BE%E8%A1%A8%E7%9A%84%E5%80%92%E6%95%B0%E7%AC%ACN%E4%B8%AA%E8%8A%82%E7%82%B9.html#%E6%80%9D%E8%B7%AF

```python
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
	head_dummy = ListNode()
	head_dummy.next = head

	slow, fast = head_dummy, head_dummy
	for _ in range(n): 
		fast = fast.next

	while(fast.next!=None):
		slow = slow.next
		fast = fast.next

	slow.next = slow.next.next
	return head_dummy.next
```

### 143. Reorder List

#### Solution: Intuitive

https://leetcode.com/problems/reorder-list/discuss/801971/Python-O(n)-by-two-pointers-w-Visualization

```python
def reorderList(self, head: Optional[ListNode]) -> None:
	"""
	Do not return anything, modify head in-place instead.
	"""
	# Save linked list in array
	arr = []
	cur, length = head, 0

	while cur:
		arr.append( cur )
		cur, length = cur.next, length + 1

	# Reorder with two-pointers
	left, right = 0, length-1
	last = head

	while left < right:
		arr[left].next = arr[right]
		left += 1

		if left == right: 
			last = arr[right]
			break

		arr[right].next = arr[left]
		right -= 1
		last = arr[left]

	if last: last.next= None
```

#### Solution: Find Middle, Reverse, and Merge

https://leetcode.com/problems/reorder-list/discuss/801883/Python-3-steps-to-success-explained

```python
def reorderList(self, head):
	#step 1: find middle
	if not head: return []
	slow, fast = head, head
	while fast.next and fast.next.next:
		slow = slow.next
		fast = fast.next.next

	#step 2: reverse second half
	prev, curr = None, slow.next
	while curr:
		nextt = curr.next
		curr.next = prev
		prev = curr
		curr = nextt    
	slow.next = None

	#step 3: merge lists
	head1, head2 = head, prev
	while head1 and head2:
		nxt1 = head1.next
		nxt2 = head2.next

		head1.next = head2
		head1 = nxt1

		head2.next = head1
		head2 = nxt2
```

## Matrix

### 73. Set Matrix Zeroes

```python
def setZeroes(self, matrix: List[List[int]]) -> None:
	"""
	Do not return anything, modify matrix in-place instead.
	"""
	m = len(matrix)
	n = len(matrix[0])
	iis = set()
	jjs = set()

	for i in range(m):
		for j in range(n):
			if matrix[i][j] == 0:
				iis.add(i)
				jjs.add(j)

	for x in iis:
		for j in range(n):
			matrix[x][j] = 0
	for i in range(m):
		for y in jjs:
			matrix[i][y] = 0
```

### 54. Spiral Matrix

注意最后剩下item的处理

```python
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
	m = len(matrix[0])
	n = len(matrix)
	res = []

	left = 0
	right = m - 1
	top = 0
	bottom = n - 1
	while left < right and top < bottom:
		for i in range(left, right):
			res.append(matrix[top][i])

		for i in range(top, bottom):
			res.append(matrix[i][right])

		for i in range(right, left, -1):
			res.append(matrix[bottom][i])

		for i in range(bottom, top, -1):
			res.append(matrix[i][left])

		top += 1
		right -= 1
		bottom -= 1
		left += 1

	if top==bottom and left == right:
		res.append(matrix[left][left])
	elif left == right and top < bottom: # n X 1 matrix
		for i in range(top, bottom+1):
			res.append(matrix[i][left])
	elif top == bottom and left < right: ## 1 X n matrix
		for i in range(left, right+1):
			res.append(matrix[top][i]) 
	return res
```

### 48. Rotate Image

first reverse up to down, then swap the symmetry 

https://leetcode.com/problems/rotate-image/discuss/18872/A-common-method-to-rotate-the-image

```python
def rotate(self, A: List[List[int]]) -> None:
	"""
	Do not return anything, modify matrix in-place instead.
	"""
	A.reverse()
	for i in range(len(A)):
		for j in range(i):
			A[i][j], A[j][i] = A[j][i], A[i][j]
```

### 79. Word Search

#### Solution: DFS + Backtracking

思路和正常dfs思路是一样的，重要的是visited的处理。在保证同一条路经每个位置只经过一次的情况下，还要保证如果经过了一个位置，但此位置不符合要求，要从visited中移除。

```python
def exist(self, board: List[List[str]], word: str) -> bool:
	m = len(board)
	n = len(board[0])

	def dfs(i,j,path):
		if path == word:
			return True
		visited.add((i,j))
		for x,y in (i+1,j),(i-1,j),(i,j+1),(i,j-1):
			if 0<=x<m and 0<=y<n and (x,y) not in visited:
				if len(path) < len(word) and board[x][y] == word[len(path)]:
					if dfs(x,y,path + board[x][y]):
						return True
		visited.remove((i,j))

	for i in range(m):
		for j in range(n):
			if board[i][j] == word[0]:
				visited = set()
				if dfs(i,j,word[0]):
					return True
	return False
```

## String

### 3. Longest Substring Without Repeating Characters

#### Solution: Sliding Window

If s[r] not in seen, we can keep increasing the window size by moving right pointer.

There are two cases if s[r] in seen:
1.s[r] is inside the current window, we need to change the window by moving left pointer to seen[s[r]] + 1.
2.s[r] is not inside the current window, we can keep increase the window

```python
def lengthOfLongestSubstring(self, s: str) -> int:
	seen = {}
	l = 0
	output = 0
	for r in range(len(s)):
		if s[r] not in seen:
			output = max(output,r-l+1)
		else:
			if seen[s[r]] < l:
				output = max(output,r-l+1)
			else:
				l = seen[s[r]] + 1
		seen[s[r]] = r
	return output
```

### 424. Longest Repeating Character Replacement

#### Solution: Sliding Window

maxf means the max frequency of the same character in the sliding window. The condition of the sliding window is j-i+1>maxf+k, which means that the current window no longer satisfied the condition and we need to move the left pointer. In the process, we also keep track of the max length which is the final answer.

```python
def characterReplacement(self, s: str, k: int) -> int:
	maxf = i = 0
	count = defaultdict(int)
	res = 0
	for j in range(len(s)):
		count[s[j]] += 1
		maxf = max(maxf, count[s[j]])
		if j - i + 1 > maxf + k:
			count[s[i]] -= 1
			i += 1
		res = max(res, j-i+1)
	return res
```

### 76. Minimum Window Substring

sliding window的关键在于什么时候扩张和什么时候缩紧，本题运用了两个Counter来跟踪当前窗口的字母和个数，并且用一个var来表示当前是否满足条件。满足的条件是：1. s中的字符c存在于t中 2. c在t中的个数和c在当前窗口下的个数一致

```python
def minWindow(self, s: str, t: str) -> str:
    counter_t = collections.Counter(t)
    counter_s = collections.defaultdict(int)

    left = 0
    satisfied = 0
    result = ""


    for right in range(len(s)):
        char = s[right]
        if char in counter_t:
            counter_s[char] += 1
            if counter_s[char] == counter_t[char]:
                satisfied += 1


        while satisfied == len(counter_t):
            if not result or (right - left + 1) < len(result):
                result = s[left:right+1]
            left_char = s[left]
            if left_char in counter_t and counter_s[left_char] == counter_t[left_char]:
                satisfied -= 1
            counter_s[left_char] -= 1
            left+=1

    return result
```

### 242. Valid Anagram

dict是可以直接比较的

```python
def isAnagram(self, s: str, t: str) -> bool:
	return Counter(s) == Counter(t)
```

### 49. Group Anagrams

利用anagram中字符全部一致的特性，用sort做common feature。

```python
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
	cnt = defaultdict(list)

	for word in strs:
		key = ''.join(sorted(word))
		cnt[key].append(word)

	return list(cnt.values())
```

### 20. Valid Parentheses

```python
def isValid(self, s: str) -> bool:
	stack = []
	for c in s:
		if c == '(' or c == '{' or c == '[':
			stack.append(c)
		if  (c == ')' or c == '}' or c == ']') and not stack:
			return False

		if c == ')' and stack:
			if stack[-1] == '(':
				stack.pop()
			else: return False
		if c == '}' and stack:
			if stack[-1] == '{':
				stack.pop()
			else: return False
		if c == ']' and stack:
			if stack[-1] == '[':
				stack.pop()
			else: return False

	if not stack:
		return True
	else:
		return False
```

### 125. Valid Palindrome

isalnum() returns True if all characters in the string are alphanumeric (either alphabets or numbers)

```python
def isPalindrome(self, s: str) -> bool:
	l, r = 0, len(s) - 1
	while l < r:
		if not s[l].isalnum():
			l += 1
		elif not s[r].isalnum():
			r -= 1
		else:
			if s[l].lower() != s[r].lower():
				return False
			else:
				l += 1
				r -= 1
	return True
```

### 5. Longest Palindromic Substring

#### Solution: DP

首先定义 P（i,j): 

P(i,j) = True s[i,j] 是回文串, P(i,j)= False s[i,j] 不是回文串

接下来 P(i,j) = (P(i+1,j−1) && S[i]==S[j])

```python
def longestPalindrome(self, s: str) -> str:
	n = len(s)
	dp = [[False] * n for _ in range(n)]

	res = ""
	for i in range(n):
		dp[i][i] = True
		res = s[i]

	for i in range(n-1,-1,-1):
		for j in range(i+1, n):
			if s[i] == s[j]:
				if j-i == 1 or dp[i+1][j-1]:
					dp[i][j] = True
					if j-i+1 > len(res):
						res = s[i:j+1]
	return res
```

#### Solution: 中心扩展

我们知道回文串一定是对称的，所以我们可以每次循环选择一个中心，进行左右扩展，判断左右字符是否相等即可。O(N^2)

```python
def longestPalindrome(self, s):
    res = ""
    for i in range(len(s)):
        # odd case, like "aba"
        tmp = self.helper(s, i, i)
        if len(tmp) > len(res):
            res = tmp
        # even case, like "abba"
        tmp = self.helper(s, i, i+1)
        if len(tmp) > len(res):
            res = tmp
    return res
 
# get the longest palindrome, l, r are the middle indexes   
# from inner to outer
def helper(self, s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1; r += 1
    return s[l+1:r]
```

### 647. Palindromic Substrings

思路跟上面的题一样

```python
def countSubstrings(self, s: str) -> int:
	n = len(s)
	res = 0
	dp = [[False] * n for _ in range(n)]

	for i in range(n):
		dp[i][i] = True
		res += 1

	for i in range(n-1,-1,-1):
		for j in range(i+1, n):
			if s[i] == s[j]:
				if j-i==1 or dp[i+1][j-1]:
					dp[i][j] = True
					res += 1
	return res
```

### 271. Encode and Decode Strings

Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Use a special sequence to denote the end of a string, and escape the sequence. Be careful of empty string and empty input as they’re different!

```python
class Codec:
    def encode(self, strs):
        """Encodes a list of strings to a single string.

        :type strs: List[str]
        :rtype: str
        """
        if len(strs) == 0:
            return ""
        else:
            return "//".join([s.replace("/", "#/#") for s in strs]) + "//"


    def decode(self, s):
        """Decodes a single string to a list of strings.

        :type s: str
        :rtype: List[str]
        """
        if len(s) == 0:
            return []
        return [seg.replace("#/#", "/") for seg in s.split("//")][:-1]
```

## Tree

### 104. Maximum Depth of Binary Tree

#### Solution: Recursive

```python
def maxDepth(self, root: Optional[TreeNode]) -> int:
	if not root:
		return 0
	return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

#### Solution: Iterative Level Order Traversal

```python
def maxDepth(self, root: Optional[TreeNode]) -> int:
	if root == None:
		return 0

	q = [root]
	res = []
	while q:
		s = len(q)
		r = []
		for _ in range(s):
			node = q.pop(0)
			r.append(node)
			if node.left: q.append(node.left)
			if node.right: q.append(node.right)
		res.append(r)
	return len(res)
```

### 100. Same Tree

```python
def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
	if not p and not q: return True
	if not p and q: return False
	if p and not q: return False
	if p and q and p.val != q.val: return False

	l = self.isSameTree(p.left, q.left)
	r = self.isSameTree(p.right, q.right)

	return l and r
```

### 226. Invert Binary Tree

```python
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
	if not root:
		return None
	root.left, root.right = root.right, root.left
	self.invertTree(root.left)
	self.invertTree(root.right)
	return root
```

### 124. Binary Tree Maximum Path Sum

```python
```
