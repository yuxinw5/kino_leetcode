## DFS & BFS

### 200. Number of Islands

```python
def numIslands(self, grid: List[List[str]]) -> int:
    m = len(grid)
    n = len(grid[0])

    def dfs(i,j):
        grid[i][j] = "2"
        for x,y in (i+1,j),(i-1,j),(i,j+1),(i,j-1):
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

### 286. Walls and Gates

You are given a m x n 2D grid initialized with these three possible values.

-1 - A wall or an obstacle.

0 - A gate.

INF - Infinity means an empty room. We use the value 2^31 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

https://zhenyu0519.github.io/2020/03/07/lc286/#leetcode-286-walls-and-gates-python

#### Solution: BFS

```python
def wallsAndGates(self, rooms):
    q = [(i, j) for i, row in enumerate(rooms) for j, r in enumerate(row) if not r]
    for i, j in q:
        for I, J in (i+1, j), (i-1, j), (i, j+1), (i, j-1):
            if 0 <= I < len(rooms) and 0 <= J < len(rooms[0]) and rooms[I][J] > 2**30:
                rooms[I][J] = rooms[i][j] + 1
                q += (I, J),
```

#### Solution: DFS

```python
def wallsAndGates(self, rooms: List[List[int]]) -> None:
    """
    Do not return anything, modify rooms in-place instead.
    """
    if not rooms:
        return []
    row = len(rooms)
    col = len(rooms[0])
    directions=[(-1,0),(0,1),(1,0),(0,-1)]
    def dfs(x,y,dis):
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0<=nx<row and 0<=ny<col and rooms[nx][ny]>rooms[x][y]:
                rooms[nx][ny]=dis+1
                dfs(nx,ny,dis+1)

    for x in range(row):
        for y in range(col):
            if rooms[x][y] == 0:
                dfs(x,y,0)
```

### 130. Surrounded Regions

要想一下从哪开始start dfs/bfs，这个题就是从边界开始想。遍历的时候mark，之后再复原，秒啊。

```python
def solve(self, board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    m = len(board)
    n = len(board[0])

    def dfs(i,j):
        board[i][j] = "."
        for x,y in (i+1,j),(i-1,j),(i,j+1),(i,j-1):
            if 0<=x<m and 0<=y<n and board[x][y] == "O":
                dfs(x,y)

    for r in range(m):
        for c in range(n):
            if (r in [0, m-1] or c in [0, n-1]) and board[r][c] == "O":
                dfs(r,c)

    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '.':
                board[i][j] = 'O'
```

### 339. Nested List Weight Sum

Given a nested list of integers, return the sum of all integers in the list weighted by their depth.

Each element is either an integer, or a list – whose elements may also be integers or other lists.

Input: [1,[4,[6]]]

Output: 27 

Explanation: One 1 at depth 1, one 4 at depth 2, and one 6 at depth 3; 1 + 4*2 + 6*3 = 27.

https://zhenyu0519.github.io/2020/03/16/lc339/

#### Solution: DFS

Weight sum is level depth times sum. We will go throught the list, if the element is digit, we sum up. If element is list, we use dfs to get into new depth and go through the new list again. The depth is start from 1.

```python
def depthSum(self, nestedList: List[NestedInteger]) -> int:      
    def DFS(nestedList, depth):
        temp_sum = 0
        for elem in nestedList:
            if elem.isInteger():
                temp_sum += elem.getInteger() * depth
            else:
                temp_sum += DFS(elem.getList(),depth+1)
        return temp_sum
    return DFS(nestedList,1)
```

#### Solution: BFS

```python
def depthSum(self, nestedList: List[NestedInteger]) -> int:
    agg = 0
    queue = collections.deque(nestedList)
    layer = 1

    while queue:
        length = len(queue)
        for _ in range(length):
            integer = queue.popleft()
            if integer.isInteger():
                agg+=layer*integer.getInteger()
            else:
                for i in integer.getList():
                    queue.append(i)
        layer+=1

    return agg
```

### 364. Nested List Weight Sum II

Given a nested list of integers, return the sum of all integers in the list weighted by their depth.

Each element is either an integer, or a list – whose elements may also be integers or other lists.

Different from the previous question where weight is increasing from root to leaf, now the weight is defined from bottom up. i.e., the leaf level integers have weight 1, and the root level integers have the largest weight.

Input: [1,[4,[6]]]

Output: 17 

Explanation: One 1 at depth 3, one 4 at depth 2, and one 6 at depth 1; 1*3 + 4*2 + 6*1 = 17.

https://zhenyu0519.github.io/2020/03/16/lc364/

```python
def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
    def dfs(nestedList, list_sum):
        temp_list = []
        for elem in nestedList:
            if elem.isInteger():
                list_sum += elem.getInteger()
            else:
                temp_list += elem.getList()
        if len(temp_list) != 0:
            list_sum+=dfs(temp_list, list_sum)
        return list_sum
    return dfs(nestedList, 0)
```

### 127. Word Ladder

求最短路径首先想到bfs。把wordlist变成set，缩短search的时间。wordlist remove已经搜索过的，防止有cycle，效果等同于visited set。

```python
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    wordList = set(wordList)
    queue = collections.deque([[beginWord, 1]])
    while queue:
        word, length = queue.popleft()
        if word == endWord:
            return length
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word in wordList:
                    wordList.remove(next_word)
                    queue.append([next_word, length + 1])
    return 0
```

### 126. Word Ladder II

好奇怪，没看明白，再看一眼

### 51. N-Queens

太妙了，哪哪都是知识点，下次自己写写

```python
def solveNQueens(self, n: int) -> List[List[str]]:
    res = []

    # check whether nth queen can be placed in that column
    def valid(nums, n):# n 是刚加进去的row no
        for i in range(n):
            if abs(nums[i]-nums[n]) == n-i or nums[i] == nums[n]:
                return False
        return True

    # nums is a one-dimension array, like [1, 3, 0, 2] means
    # first queen is placed in column 1, second queen is placed
    # in column 3, etc.
    def dfs(nums, row_insex, path, res):
        if row_insex == len(nums):
            res.append(path)
            return
        for i in range(len(nums)):
            nums[row_insex] = i
            if valid(nums, row_insex):
                tmp = "."*len(nums)
                dfs(nums, row_insex+1, path+[tmp[:i]+"Q"+tmp[i+1:]], res)

    dfs([-1]*n, 0, [], res)
    return res
```

## 单点BFS

### 690. Employee Importance

注意题目中的数据结构，employees[i]并不代表id为i的employee

```python
def getImportance(self, employees: List['Employee'], id: int) -> int:
    res = 0
    def dfs(i):
        nonlocal res
        res += emps[i].importance
        for j in emps[i].subordinates:
            dfs(j)

    emps= {emp.id: emp for emp in employees}
    dfs(id)
        return res
```

### 1236. Web Crawler

https://leetcode.jp/leetcode-1236-web-crawler-%E8%A7%A3%E9%A2%98%E6%80%9D%E8%B7%AF%E5%88%86%E6%9E%90/

```python
def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
    visited = {startUrl}
    queue = collections.deque([startUrl])
    host_name = 'http://' + startUrl.split('/')[2]

    while queue:
        now = queue.popleft()
        for new in htmlParser.getUrls(now):
            if new.startswith(host_name) and new not in visited:
                queue.append(new)
                visited.add(new)

    return list(visited)
```

## 多点BFS

### 994. Rotting Oranges

传统bfs layers的操作，把steps的信息包含在queue的items里面。本题还用了count来判断是否全部被感染，很妙，不用最后再loop一遍查看了。

```python
def orangesRotting(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    q = collections.deque([])
    cnt = 0

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                q.append((i,j,0))
            if grid[i][j] == 1:
                cnt += 1

    res = 0
    while q:
        i, j, steps = Q.popleft()
        res = steps
        for x, y in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
            if 0<=x<m and 0<=y<n and grid[x][y] == 1:
                grid[x][y] = 2
                cnt -= 1
                q.append((x,y, steps + 1))

    return res if cnt == 0 else -1
```

### 542. 01 Matrix

这个题某种程度上揭露了BFS找最短路径的本质，他其实不是找到最短路径，而是遍历的一种方式，只不过在过程中是层层搜索，遇到曾经搜索过的node就说明肯定不需要再更新，因为在此之前的路径一定要小。

所以这个题要记得先把默认路径值设为inf，每层都比较当前路径长度，遇到小的才update

```python
def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
    m = len(mat)
    n = len(mat[0])

    q = []

    for i in range(m):
        for j in range(n):
            if mat[i][j] == 0:
                q.append((i,j,0))
            else:
                mat[i][j] = 0x7fffffff

    while q:
        i,j,dis = q.pop(0)
        for x,y in (i+1,j),(i-1,j),(i,j+1),(i,j-1):
            if 0<=x<m and 0<=y<n and mat[x][y] > dis:
                mat[x][y] = dis+1
                q.append((x,y,dis+1))
    return mat
```

### 1162. As Far from Land as Possible

跟上一题原理基本一致，只不过在找max的时候要注意1可能表示距离也可能表示原本的item

```python
def maxDistance(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])

    q = []
    meet = False

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0:
                grid[i][j] = inf
                meet = True
            else:
                q.append((i,j,0))

    while q:
        i,j,dis = q.pop(0)
        for x,y in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
            if 0<=x<m and 0<=y<n and grid[x][y] > dis+1:
                grid[x][y] = dis+1
                q.append((x,y,dis+1)) 

    res = max(max(grid[i]) for i in range(m))
    if not meet or res == inf:
        return -1
    return res
```

## 层数BFS

### 815. Bus Routes

这个题真的很妙，layers不是特别明显，所以主要是搞清楚layer代表的是什么。因为我们要求bus数最少，所以layer就是bus数。

那么怎样把layers连接起来呢，就需要把all of the stops you can reach for taking one time of bus记录下来。

This is a very good BFS problem. In BFS, we need to traverse all positions in each level firstly, and then go to the next level.

Our task is to figure out:
1. What is the level in this problem?
2. What is the position we want in this problem?
3. How to traverse all positions in a level?

For this problem:
1. The level is each time to take bus.
2. The position is all of the stops you can reach for taking one time of bus.
3. Using a queue to record all of the stops can be arrived for each time you take buses.

```python
def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
    #  record stop to routes
    to_routes = collections.defaultdict(set)
    for i, route in enumerate(routes):
        for j in route:
            to_routes[j].add(i)

    q = [(source, 0)]
    seen_stops = set([source])
    seen_routs = set()

    while q:
        stop, bus = q.pop(0)
        if stop == target: 
            return bus

        for i in to_routes[stop]:
            if i not in seen_routs:
                seen_routs.add(i)
                for j in routes[i]:
                    if j not in seen_stops:
                        q.append((j, bus + 1))
                        seen_stops.add(j)
    return -1
```

### 1091. Shortest Path in Binary Matrix

没什么特别的，只不过是8个方向而已。

```python
def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
    if grid[-1][-1]==1 or grid[0][0]==1:
        return -1
    n = len(grid)
    q = [(0, 0, 1)]
    grid[0][0] = 1

    while q:
        i,j,d = q.pop(0)
        if i == n-1 and j == n-1: 
            return d
        for x,y in ((i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)):
            if 0<=x<n and 0<=y<n and grid[x][y] == 0:
                grid[x][y] = 1
                q.append((x, y, d+1))
    return -1
```

### 773. Sliding Puzzle

不会，有空看

### 934. Shortest Bridge

Bloomberg面试题啊啊啊啊，好好看！

其实思路蛮直接的，先用dfs区分出两个岛，然后对一个岛做multi source bfs，一旦找到另一个岛就停止。

```python
def shortestBridge(self, grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0])
    q = []

    def dfs(i,j):
        q.append((i,j,0))
        grid[i][j] = 2
        for (x,y) in (i+1,j), (i-1,j), (i,j+1), (i,j-1):
            if 0<=x<m and 0<=y<n and grid[x][y] == 1:
                dfs(x,y)
    flag = False
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and not flag:
                dfs(i,j)
                flag = True

    while q:
        i,j,d = q.pop(0)
        for (x,y) in (i+1,j), (i-1,j), (i,j+1), (i,j-1):
            if 0<=x<m and 0<=y<n:
                if grid[x][y] == 0:
                    q.append((x,y,d+1))
                    grid[x][y] = 2
                elif grid[x][y] == 1:
                    return d
    return -1
```

### 279. Perfect Squares

#### Solution: DP

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

#### Solution: BFS

一开始很难想到这个思路，看看就好，还是dp比较清晰

The root node is n, and we are trying to keep reduce a perfect square number from it each layer. So the next layer nodes are {n - i**2 for i in range(1, int(n**0.5)+1)}. And target leaf node is 0, indicates n is made up of a number of perfect square numbers and depth is the least number of perfect square numbers.

```python
def numSquares(self, n):
	squares = [i**2 for i in range(1, int(n**0.5)+1)]
	d, q, nq = 1, {n}, set()
	while q:
		for node in q:
			for square in squares:
				if node == square: return d
				if node < square: break
				nq.add(node-square)
		q, nq, d = nq, set(), d+1
```
