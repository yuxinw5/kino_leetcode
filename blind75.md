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

