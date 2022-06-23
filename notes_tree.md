## Tree

### 144.Binary Tree Preorder Traversal

#### Solution: Recursive

```python
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    res = []
    def pre(root):
        if not root:
            return
        res.append(root.val)
        pre(root.left)
        pre(root.right)
    pre(root)
    return res
```

#### Solution: Iterative

将左右子树分别压栈，然后每次从栈里取元素。需要注意的是，因为我们应该先访问左子树，而栈的话是先进后出，所以我们压栈先压右子树

```python
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
        return 
    res = []
    stack = [root]
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return res
```

### 94. Binary Tree Inorder Traversal

#### Solution: Recursive

```python
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    res = []
    def ino(root):
        if not root:
            return
        ino(root.left)
        res.append(root.val)
        ino(root.right)
    ino(root)
    return res
```

#### Solution: Iterative

中序遍历是左中右，先访问的是二叉树顶部的节点，然后一层一层向下访问，直到到达树左面的最底部，再开始处理节点（也就是在把节点的数值放进result数组中），这就造成了处理顺序和访问顺序是不一致的。

那么在使用迭代法写中序遍历，就需要借用指针的遍历来帮助访问节点，栈则用来处理节点上的元素。

```python
def inorderTraversal(self, root: TreeNode) -> List[int]:
    if not root:
        return []
    stack = []  # 不能提前将root结点加入stack中
    result = []
    cur = root
    while cur or stack:
        # 先迭代访问最底层的左子树结点
        if cur:     
            stack.append(cur)
            cur = cur.left		
        # 到达最左结点后处理栈顶结点    
        else:		
            cur = stack.pop()
            result.append(cur.val)
            # 取栈顶元素右结点
            cur = cur.right	
    return result
```

### 145. Binary Tree Postorder Traversal

#### Solution: Recursive

```python
def postorderTraversal(self, root: TreeNode) -> List[int]:
    result = []

    def traversal(root: TreeNode):
        if root == None:
            return
        traversal(root.left)    # 左
        traversal(root.right)   # 右
        result.append(root.val) # 后序

    traversal(root)
    return result
```

#### Solution: Iterative

前序遍历的顺序是 根 -> 左 -> 右，左右其实是等价的，所以我们也可以轻松的写出 根 -> 右 -> 左 的代码。

后序遍历的顺序是 左 -> 右 -> 根。我们可以按照 根 -> 右 -> 左遍历，最后逆序。

```python
def postorderTraversal(self, root: TreeNode) -> List[int]:
    if root is None:
        return []

    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left: 
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    result.reverse()        
    return result
```

### 102. Binary Tree Level Order Traversal

```python
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return root
    q = [root]
    res = []
    while q:
        s = len(q)
        r = []
        for _ in range(s):
            n = q.pop(0)
            r.append(n.val)
            if n.left: q.append(n.left)
            if n.right: q.append(n.right)
        res.append(r)
    return res
```

## Preorder

### 100. Same Tree

注意不同情况要考虑全面

```python
def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    if not p and not q:
        return True
    if not p or not q: # one of p and q is None
        return False
    if p.val!=q.val:
        return False
    return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

### 101. Symmetric Tree

注意是从里到外一一对应相等，而不是每个subtree都是mirror。也可以用level order做

```python
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    if not root:
        return True
    return self.helper(root.left, root.right)

def helper(self, l, r):
    if not l and not r:
        return True
    if not l or not r:
        return False
    if l.val!=r.val:
        return False
    return self.helper(l.left, r.right) and self.helper(l.right, r.left)
```

### 226. Invert Binary Tree

跟上一题做个对比

```python
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    root.left, root.right = root.right, root.left
    self.invertTree(root.left)
    self.invertTree(root.right)
    return root
```

### 257. Binary Tree Paths

```python
def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
    def dfs(cur,path):
        path += str(cur.val)
        if not cur.left and not cur.right:
            res.append(path)

        if cur.left:
            dfs(cur.left, path + '->')
        if cur.right:
            dfs(cur.right, path + '->')

    res = []
    dfs(root,"")
    return res
```

### 112. Path Sum

```python
def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
    if not root:
        return False

    if not root.left and not root.right and root.val == targetsum:
        return True

    return self.hasPathSum(root.left, targetsum-root.val) or self.hasPathSum(root.right, targetsum-root.val)
```

### 113. Path Sum II

注意edge cases，比如虽然check了left right是否为空之后再进行下一步递归，但有可能root最开始就是空的。

```python
def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
    res = []
    if not root:
        return []
    def dfs(node, path):
        if sum(path) + node.val == targetSum and not node.left and not node.right:
            res.append(path + [node.val])
        if node.left:
            dfs(node.left, path + [node.val])
        if node.right:
            dfs(node.right, path + [node.val])
    dfs(root, [])
        return res
```

### 129. Sum Root to Leaf Numbers

访问当前node的时候就在path加上val，或者访问下一个node的时候加val，两个方法都可以，重要的是append result时候注意就好。可以对比上一题的方法。

```python
def sumNumbers(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    res = 0
    def dfs(root, path):
        nonlocal res
        path += str(root.val)
        if not root.left and not root.right:
            res += int(path)
        if root.left:
            dfs(root.left, path)
        if root.right:
            dfs(root.right, path)
    dfs(root, "")
    return res
```

### 298. Binary Tree Longest Consecutive Sequence

Given a binary tree, find the length of the longest consecutive sequence path.

The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The longest consecutive path need to be from parent to child (cannot be the reverse).

```python
def longestConsecutive(self, root):
    if not root: 
        return 0
    res = 1
    def dfs(root, curLen):
        nonlocal res
        res = max(curLen, res)
        if root.left:
            if root.left.val == root.val + 1:
                dfs(root.left, curLen + 1)
            else:
                dfs(root.left, 1)
        if root.right:
            if root.right.val == root.val + 1:
                dfs(root.right, curLen + 1)
            else:
                dfs(root.right,1)

    dfs(root, 1)
    return res
```

### 111. Minimum Depth of Binary Tree

典型BFS，layer search，一旦找到符合的就停止，it is guaranteed to be the shortest。

当然这个题用普通recursive也可以。

```python
def minDepth(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0

    q = [(root, 1)]

    while q:
        node, layer = q.pop(0)
        if not node.left and not node.right:
            return layer
        if node.left:
            q.append((node.left, layer+1))
        if node.right:
            q.append((node.right, layer+1))
```

### 114. Flatten Binary Tree to Linked List

可以手动找到left tree的tail，并不一定要return什么

```python
def flatten(self, root: Optional[TreeNode]) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    if not root:
        return 

    self.flatten(root.left)
    self.flatten(root.right)

    tail = root.left
    if tail:
        while tail and tail.right:
            tail = tail.right
        tail.right = root.right
        root.right = root.left
        root.left = None
```

## Postorder

### 104. Maximum Depth of Binary Tree

简单的递归，极致的享受

```python
def maxDepth(self, root: Optional[TreeNode]) -> int:
    if root == None:
        return 0

    return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

### 110. Balanced Binary Tree

#### Worse Case O(nlogn) Solution

最差的情况是左右两边除了一边最下面一个不平衡其他都平衡

```python
def isBalanced(self, root: TreeNode) -> bool:
    if not root:
        return True
    left_height = self.height(root.left)
    right_height = self.height(root.right)

    return abs(left_height-right_height)<2 and self.isBalanced(root.left) and self.isBalanced(root.right)

def height(self, root):
    if not root:
        return 0
    return max(1+self.height(root.left), 1+self.height(root.right))
```

#### O(N) Solution

可以直接在算height的时候判断是否balance

```python
def isBalanced(self, root):
    self.balanced = True
    def height(root):
      if not root or not self.balanced: return -1
      l = height(root.left)
      r = height(root.right)
      if abs(l - r) > 1:
        self.balanced = False
        return -1
      return max(l, r) + 1
    height(root)
    return self.balanced
```

### 124. Binary Tree Maximum Path Sum

https://leetcode.com/problems/binary-tree-maximum-path-sum/discuss/603423/Python-Recursion-stack-thinking-process-diagram

还是要复习思路

### 250. Count Univalue Subtrees

Given the root of a binary tree, return the number of uni-value subtrees.

A uni-value subtree means all nodes of the subtree have the same value.

#### Solution: Bottom Up

Just check the whether the left and right subtrue are uni-value or not

If left and right subtruee are uni-vale, then check whether the node value and the value of two subtrees are the same.

```python
def countUnivalSubtrees(self, root: TreeNode) -> int:
        self.res = 0
        self.helper(root)
        return self.res
        
def helper(self, node):
    if node:
        if not node.left and not node.right:
            self.res += 1
            return node.val
        if node.left and node.right:
            left = self.helper(node.left)
            right = self.helper(node.right)
            if left!= node.val or right!=node.val:
                return None
        elif node.left:
            left = self.helper(node.left)
            if left!= node.val:
                return None
        else:
            right = self.helper(node.right)
            if right!= node.val:
                return None

        self.res+=1
        return node.val
```

### 366. Find Leaves of Binary Tree

### 337. House Robber III

### 107. Binary Tree Level Order Traversal II

```python
def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    q = [root]
    res = []
    while q:
        s = len(q)
        r = []
        for _ in range(s):
            node = q.pop(0)
            r.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(r)
    return res[::-1]
```

### 103. Binary Tree Zigzag Level Order Traversal

```python
def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    q = [root]
    res = []
    reverse = 0
    while q:
        s = len(q)
        r = []
        for _ in range(s):
            node = q.pop(0)
            r.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        if not reverse:
            res.append(r)
            reverse = 1
        else:
            res.append(r[::-1])
            reverse = 0
    return res
```

### 199. Binary Tree Right Side View

#### Solution: Level Order

```python
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    q = [root]
    res = []
    while q:
        s = len(q)
        for i in range(s):
            node = q.pop(0)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
            if i == s-1:
                res.append(node.val)
        return res
```

#### Solution: DFS

可以简单看一下，通过现有result和level的关系决定选取哪个node

```python
def rightSideView(self, root: TreeNode) -> List[int]:
    res = []
    self.dfs(root, 1, res)
    return res

def dfs(self, node, level, res):
    if not node:
        return
    if len(res)<level:
        res.append(node.val)
    self.dfs(node.right, level+1, res)
    self.dfs(node.left, level+1, res)
```
