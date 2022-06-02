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

#### Solution: Binary Search

1. loop is left < right, which means inside the loop, left always < right
2. since we use round up for mid, and left < right from (1), right would never be the same as mid
3. Therefore, we compare mid with right, since they will never be the same from (2)
4. if nums[mid] < nums[right], we will know the minimum should be in the left part, so we are moving right.
5. if nums[mid] > nums[right], minimum should be in the right part, so we are moving left. Since nums[mid] > nums[right],mid can't be the minimum, we can safely move left to mid + 1, which also assure the interval is shrinking

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
