## Graph

## 基础

### 133. Clone Graph

经典题目了，mapping既来做新旧图的mapping，又做visited set。

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

### 399. Evaluate Division

这个题值得一看，需要注意的有两个点。第一个注意已知a/b=2和b/c=3，那么a/c=(a/b)*(b/c)，转换的时候是相乘的关系。

第二点注意dfs什么情况需要return，if vv是指vv不是None，因此dfs最后无需return。但如果最后return -1的话，if vv要改成if vv != -1

```python
def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    graph = defaultdict(list)
    for i in range(len(equations)):
        a = equations[i][0]
        b = equations[i][1]
        graph[a].append((b,values[i]))
        graph[b].append((a,1/values[i]))

    def dfs(start,end, value):
        if start not in graph or end not in graph:
            return -1
        visited.add(start)

        if start == end:
            return value

        for n in graph[start]:
            node = n[0]
            v = n[1]
            if node not in visited:
                vv = dfs(node,end,v *value)
                if vv: 
                    return vv

    res = []
    for i in queries:
        visited = set()
        temp =  dfs(i[0],i[1],1)
        res.append(-1 if not temp else temp)
    return res
```

### 310. Minimum Height Trees

方法叫leaves removal，但本质就是level order traversal/Topological。queue里存当前leaves，每层都要清空res，知道queue为空，我们就知道到最后一层了。

```python
def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
    if n == 1: return [0] 
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    q = [i for i in range(n) if len(adj[i]) == 1]

    res = []

    while q:
        s = len(q)
        res = []
        for _ in range(s):
            node = q.pop(0)
            res.append(node)
            for n in adj[node]:
                adj[n].remove(node)
                if len(adj[n]) == 1:
                    q.append(n)
    return res
```

## Union Find

### 261. Graph Valid Tree

BFS/DFS的做法在BLIND里面都做过了，下面是Union Find的做法。

https://www.goodtecher.com/leetcode-261-graph-valid-tree/

Use the union-find to group nodes. If the graph is a tree, it should have only one connected component and also the number of nodes – 1 should be equal to the number of edges.

```python
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if n - 1 != len(edges):
            return False
        
        self.father = [i for i in range(n)]
        self.size = n
        
        for a, b in edges:
            self.union(a, b)
            
        return self.size == 1
    
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        
        if root_a != root_b:
            self.size -= 1
            self.father[root_a] = root_b
            
    def find(self, node):
        path = []
        
        while node != self.father[node]:
            path.append(node)
            node = self.father[node]
            
        for n in path:
            self.father[n] = node
        return node
```

### 323. Number of Connected Components in an Undirected Graph

在BLIND里也做过了，union find方法跟上面一题是一样的，求size就好了。

### 305. Number of Islands II

题目： https://goodtecher.com/leetcode-305-number-of-islands-ii/

https://blog.csdn.net/qq_37821701/article/details/108414572

Whenever we make a position to island, we also use union-find to link its neighbors so that the next time if we visit any of neighbor positions, we know it belongs to the same island.

```python
class UnionFind:
    def __init__(self):
        self.father = {}
        self.count = 0
        
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
    
        if root_a != root_b:
            self.father[root_b] = root_a
            self.count -= 1
        
    def find(self, point):
        path = []
        while point != self.father[point]:
            path.append(point)
            point = self.father[point]
        
        for p in path:
            self.father[p] = point
        return point  
    
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        islands = set()
        results = []
        union_find = UnionFind()
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        
        for position in positions:
            x, y = position[0], position[1]
            if (x, y) in islands:
                results.append(union_find.count)
                continue
                
            islands.add((x, y))
            union_find.father[(x, y)] = (x, y)
            union_find.count += 1
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) in islands:
                    union_find.union((x, y), (nx, ny))
                    
            results.append(union_find.count)
        return results   
```

## 图形学

### 149. Max Points on a Line

主要就是按照斜率作为key，遇到一直的点就加一，如果是相同的点就直接剪枝操作了。

```python
def maxPoints(self, points: List[List[int]]) -> int:
    l = len(points)
    m = 0
    for i in range(l):
        dic = {'i': 1}
        same = 0
        for j in range(i+1, l):
            tx, ty = points[j][0], points[j][1]
            if tx == points[i][0] and ty == points[i][1]: 
                same += 1
                continue
            if points[i][0] == tx: 
                slope = 'i'
            else:
                slope = (points[i][1]-ty) * 1.0 /(points[i][0]-tx)
            if slope not in dic: 
                dic[slope] = 1
            dic[slope] += 1
        m = max(m, max(dic.values()) + same)
    return m
```

## Trie

208，211，212，BLIND里面都包含了，自己去看吧

## Topological Sort

### 207. Course Schedule

#### Solution: Topological Sort

这题跟下面一题几乎一样，除了return的那一行。建议先看下面一题更有代表性。

```python
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    G, indegree, q, ans = defaultdict(list), [0]*numCourses, deque(), []
    for nxt, pre in prerequisites:
        G[pre].append(nxt)
        indegree[nxt] += 1

    for i in range(numCourses):
        if indegree[i] == 0:
            q.append(i)

    while q:
        cur = q.popleft()
        ans.append(cur)
        for nextCourse in G[cur]:
            indegree[nextCourse] -= 1
            if indegree[nextCourse] == 0: 
                q.append(nextCourse)

    return True if len(ans) == numCourses else False
```

#### Solution: DFS Cycle Detection

注意循环里面call hasCycle之后一定要立刻return，不然他会继续向下走，就会return错误结果

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

### 210. Course Schedule II

#### Solution: Topological + BFS

题目跟310有一点点像，可以去看看，Topo的主要方法论就是：

1. 建立adj graph 
2. 建立indegree dict 
3. find the Node has 0 inDegree. (If none, there must be a circle)
4. 每次找到indegree是0的都append

这题还有一个关键就是Cycle Detection：

return ans if all courses are finished. If it is impossible to finish all courses, we return empty array. This case will occur only when the graph formed contains a cycle. It can be proved that a solution will always exist for a DAG.

```python
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    G, indegree, q, ans = defaultdict(list), [0]*numCourses, deque(), []
    for nxt, pre in prerequisites:
        G[pre].append(nxt)
        indegree[nxt] += 1

    for i in range(numCourses):
        if indegree[i] == 0:
            q.append(i)

    while q:
        cur = q.pop(0)
        ans.append(cur)
        for nextCourse in G[cur]:
            indegree[nextCourse] -= 1
            if indegree[nextCourse] == 0: 
                q.append(nextCourse)

    return ans if len(ans) == numCourses else []
```

#### Solution: DFS

注意append到res的时机很重要，要完全visited完毕之后才行。

```python
def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    requirement = collections.defaultdict(list)
    for course, pre_course in prerequisites:
        requirement[course].append( pre_course )

    visited = set()
    visiting = set()
    out = []
    def dfs(course):
        if course in visited: return True
        if course in visiting: return False

        visiting.add(course)
        for pre in requirement[course]:
            if not dfs(pre):
                return False
        visiting.remove(course)
        visited.add(course)
        out.append(course)
        return True

    for c in range(numCourses):
        if not dfs(c):
            return []
    return out
```

### 269. Alien Dictionary

https://www.lintcode.com/problem/892/

注意点1: 如果字符串 a 是字符串 b 的前缀，且 b 出现在 a 之前，那么这个顺序是无效的。

注意点2: 如何建立indegree和graph。我们只能通过word出现的顺序和char出现的顺序得到信息，所以只能从前向后比较。在word level每次只比较挨着的两个，这样是有效的，因为从前向后已经建立起了顺序。并且在char level一旦找到不一样的char就要立刻停止，因为之后的信息是没有意义的。

注意点3: 用heap来满足同一等级char按照人类字典序排序，heapq compares values from the queue using using the "less than" operator regardless of the type of the value。char也是可以比较的

```python
def alien_order(self, words: List[str]) -> str:
        in_degree = {ch:0 for w in words for ch in w}
        graph = {ch:[] for w in words for ch in w}

        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            if len(w1)>len(w2) and w1[:len(w2)]==w2: 
                return ''   # check if w1 is the prefix of w2
            for j in range(min(len(w1), len(w2))):
                if w1[j] != w2[j]:
                    in_degree[w2[j]] += 1
                    graph[w1[j]].append(w2[j])
                    break

        q = sorted([i for i in in_degree if in_degree[i]==0])
        res = []
        while q:
            ch = heappop(q)
            res.append(ch)
            for n in graph[ch]:
                in_degree[n] -= 1
                if in_degree[n]==0:
                    heappush(q, n) 
            
        res = ''.join(res)
        if len(res)!=len(in_degree):
            res = ''

        return res
```
