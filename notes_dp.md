## DP

## 一维

### 70. Climbing Stairs

这道题本质是 fibonacci sequence，只是F1, F2从1，1开始

Fibonacci sequence:
1，1，2，3，5，8，13，21，……。即数列满足递推公式![[公式]](https://www.zhihu.com/equation?tex=+F_%7Bn%2B2%7D+%3D+F_%7Bn%2B1%7D+%2B+F_%7Bn%7D)，![[公式]](https://www.zhihu.com/equation?tex=F_1+%3D+F_2+%3D+1)

```python
def climbStairs(self, n: int) -> int:
      if n == 1 or n == 2:
          return n
      dp = [0] * (n+1)
      dp[1] = 1
      dp[2] = 2
      for i in range(3, n+1):
          dp[i] = dp[i-1] + dp[i-2]

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

### 63. Unique Paths II

处理obstacle的时候要小心一点，包括起始点也要考虑.

```python
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
      m = len(obstacleGrid)
      n = len(obstacleGrid[0])
      dp = [[0 for i in range(n)] for j in range(m)]
      dp[0][0] = 1 if obstacleGrid[0][0] != 1 else 0
      for i in range(1, n):
          if obstacleGrid[0][i] != 1:
              dp[0][i] = dp[0][i-1]

      for i in range(1, m):
          if obstacleGrid[i][0] != 1:
              dp[i][0] = dp[i-1][0]

      for i in range(1,m):
          for j in range(1,n):
              if obstacleGrid[i][j] == 1:
                  dp[i][j] = 0
                  continue
              dp[i][j] = dp[i-1][j] + dp[i][j-1]

      return dp[-1][-1]
```

### 279. Perfect Squares

完全平方数就是物品（可以无限件使用），凑个正整数n就是背包，问凑满这个背包最少有多少物品.

dp[j]：和为j的完全平方数的最少数量为dp[j]

```python
def numSquares(self, n: int) -> int:
      nums = [i**2 for i in range(1, n + 1) if i**2 <= n]
      dp = [n + 1] * (n + 1)
      dp[0] = 0

      for num in nums:
          for j in range(num, n + 1):
              dp[j] = min(dp[j], dp[j - num] + 1)
      return dp[n]
```

### 139. Word Break

注意这种第一个index也要单独考虑的，一般要设置一个dummy。 dp[i]：string[:i]是否符合条件（注意不包括i，这样dp0设置为True之后方便判断第一个char是否在worddict里面）

```python
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
      dp = [False for _ in range(len(s)+1)]
      dp[0] = True
      for i in range(1, len(s)+1):
          for j in range(i):
              if s[j:i] in wordDict and dp[j]:
                  dp[i] = True
      return dp[-1]
```

### 375. Guess Number Higher or Lower II

worst case is that you will take as many steps as it can to know the correct answer. But you should wisely(which means to minimize) pick a number thus make your spending as less as possible.

https://leetcode.com/problems/guess-number-higher-or-lower-ii/discuss/84766/Clarification-on-the-problem-description.-Problem-description-need-to-be-updated-!!!

1. 根据递推公式确定是bottom up，所以注意l的遍历顺序是倒叙
2. dp[l][r]表示l到r所需最少的钱，pick pivot的时候边界都可以被取到，所以dp长度是n+2 因为加了两个padding

```python
def getMoneyAmount(self, n: int) -> int:
      dp = [[0] * (n + 2) for _ in range(n + 2)]
      for l in range(n, 0, -1):
          for r in range(l + 1, n + 1):
              dp[l][r] = math.inf
              for pick in range(l, r + 1):
                  leftCost = dp[l][pick - 1] + pick  # Cost on the left side, if pivot is not a secret number
                  rightCost = dp[pick + 1][r] + pick  # Cost on the right side, if pivot is not a secret number
                  cost = max(leftCost, rightCost)  # The cost is the maximum between the left side and the right side
                  dp[l][r] = min(dp[l][r], cost)  # Choose pivot which will cause minimum cost
      return dp[1][n]
```

### 312. Burst Balloons

不会

### 322. Coin Change

典型完全背包，dp长度是amount+1是为了配合index的匹配

dp[j]：凑足总额为j所需钱币的最少个数为dp[j]

```python
def coinChange(self, coins: List[int], amount: int) -> int:           
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0

    for c in coins:
        for j in range(c, amount+1):
            dp[j] = min(dp[j], dp[j-c] + 1)

    return dp[amount] if dp[amount] < amount + 1 else -1
```

## 二维

### 256. Paint House

### 265. Paint House II

### 64. Minimum Path Sum

没什么好说的，注意边界处理，这个题甚至不用extra space，直接原grid edit

```python
def minPathSum(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])

    for i in range(1,n):
      grid[0][i] += grid[0][i-1]

    for i in range(1,m):
      grid[i][0] += grid[i-1][0]

    for i in range(1,m):
      for j in range(1,n):
          grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[-1][-1]
```

### 72. Edit Distance

Let dp(i, j) is the minimum number of operations required to convert s1[0..i-1] string to s2[0..j-1] string.

To compute dp(i, j):

Base case:

1. If i == 0 then we need to insert j chars to convert "" into s2[0..j-1]
2. If j == 0 then we need to delete i chars to convert s1[0..i-1] into ""

If s1[i-1] == s2[j-1] then dp(i, j) = dp(i-1, j-1)

Else: Choose the minimum cost among 3 operators

1. Delete: dp(i-1, j) + 1
2. Insert: dp(i, j-1) + 1
3. Replace: dp(i-1, j-1) + 1

```python
def minDistance(self, word1: str, word2: str) -> int:
    dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
    for i in range(len(word1)+1):
        dp[i][0] = i
    for j in range(len(word2)+1):
        dp[0][j] = j
    for i in range(1, len(word1)+1):
        for j in range(1, len(word2)+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
    return dp[-1][-1]
```

### 97. Interleaving String

#### Solution: DP

dp[i][j] represents isInterleave(s1[:i], s[:j], s3[:i+j])

我们定义一个 boolean 二维数组 dp [ i ] [ j ] 来表示 s1[ 0, i ) 和 s2 [ 0, j ） 组合后能否构成 s3 [ 0, i + j )，注意不包括右边界，主要是为了考虑开始的时候如果只取 s1，那么 s2 就是空串，这样的话 dp [ i ] [ 0 ] 就能表示 s2 取空串。这个dummy设置的真的很妙

最开始想复杂了，其实不需要循环之前的index看能不能组成，只需要看上一个就可以了。

```python
def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
    if len(s1)+len(s2)!=len(s3):
        return False

    dp = [[False]*(len(s1)+1) for _ in range(len(s2)+1)]
    dp[0][0] = True

    for j in range(1, len(dp[0])):
        dp[0][j] = s1[j-1]== s3[j-1]
        if not dp[0][j]:
            break

    for i in range(1, len(dp)):
        dp[i][0] = s2[:i]==s3[:i]
        if not dp[i][0]:
            break

    for i in range(1, len(dp)):
        for j in range(1, len(dp[0])):
            dp[i][j] = (dp[i-1][j] and s2[i-1]==s3[i+j-1]) or (dp[i][j-1] and s1[j-1]==s3[i+j-1])

    return dp[-1][-1]
```

### 174. Dungeon Game

