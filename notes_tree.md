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

Given a binary tree, collect a tree’s nodes as if you were doing this: Collect and remove all leaves, repeat until the tree is empty.

Use DFS to find the height of the subtree, append all roots with same height in a same list and return the result

```python
def findLeaves(self, root):
    def dfs(root):
        if not root:
            return -1
        depth = max(dfs(root.left), dfs(root.right))+1
        if depth == len(res):
            res.append([])
        res[depth].append(root.val)
        return depth
    res = []
    dfs(root)
    return res
```

### 337. House Robber III

Redefine rob(root) as a new function which will return an array of two elements, the first element of which denotes the maximum amount of money that can be robbed if root is not robbed, while the second element signifies the maximum amount of money robbed if it is robbed.

https://leetcode.com/problems/house-robber-iii/discuss/79330/Step-by-step-tackling-of-the-problem

```python

```

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

## BST

### 98. Validate Binary Search Tree

BST特性，中序遍历是有序数组。

```python
def isValidBST(self, root: Optional[TreeNode]) -> bool:
    res = []
    def inorder(root):
        if not root:
            return
        inorder(root.left)
        res.append(root.val)
        inorder(root.right)

    inorder(root)
    if len(res) <= 1:
        return True
    for i in range(1, len(res)):
        if res[i] <= res[i-1]:
            return False
    return True
```

可以继续优化space complexity

Return the smallest and the largest value in the subtree, one node's value must be larger than the max in left subtree and must be smaller than the min in the right subtree

```python
def isValidBST(self, root: TreeNode) -> bool:
    if not root:
        return True
    self.res = True
    self.dfs(root)
    return self.res

def dfs(self, node):
    if not node.left and not node.right:
        return node.val, node.val
    if not self.res:
        return 0,0
    val1 = node.val
    val2 = node.val
    if node.left:
        left1, left2 = self.dfs(node.left) # left1 is the small one
        val1 = left1
        if not node.val>left2:
            self.res = False
    if node.right:
        right1, right2 = self.dfs(node.right)
        val2 = right2
        if not node.val<right1:
            self.res = False

    return val1, val2
```

### 235. Lowest Common Ancestor of a Binary Search Tree

利用好BST特性，并且如果找到一个节点，发现左子树出现结点p，右子树出现节点q，或者 左子树出现结点q，右子树出现节点p，那么该节点就是节点p和q的最近公共祖先。

```python
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    if root.val > p.val and root.val > q.val:
        return self.lowestCommonAncestor(root.left, p, q)
    if root.val < p.val and root.val < q.val:
        return self.lowestCommonAncestor(root.right, p, q)
    return root
```

### 108. Convert Sorted Array to Binary Search Tree

```python
def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
    if not nums: return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])

    root.left = self.sortedArrayToBST(nums[:mid])
    root.right = self.sortedArrayToBST(nums[mid+1:])

    return root
```

### 109. Convert Sorted List to Binary Search Tree

#### Solution: 转化为list 

详见上一题

#### Solution: Two Pointers

快慢指针找到middle point，并且recursive call的时候注意切割list

```python
def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
    if not head:
        return 
    if not head.next:
        return TreeNode(head.val)

    slow, fast = head, head.next.next
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    tmp = slow.next

    slow.next = None
    root = TreeNode(tmp.val)
    root.left = self.sortedListToBST(head)
    root.right = self.sortedListToBST(tmp.next)
    return root
```

### 173. Binary Search Tree Iterator

就是94的迭代版本，值得好好看一下

```python
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.root = root
        self.stack = []

    def next(self) -> int:
        while self.root:
            self.stack.append(self.root)
            self.root = self.root.left
        top = self.stack.pop()
        self.root = top.right
        return top.val


    def hasNext(self) -> bool:
        return self.root or self.stack
```

### 230. Kth Smallest Element in a BST

k is a global var because we need to keep track of the number we have counted, so we cannot pass k in the funtion as a parameter.

```python
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    self.k = k
    self.res = None

    def inorder(node):
        if not node:
            return
        inorder(node.left)
        self.k -= 1
        if self.k == 0:
            self.res = node.val
            return
        inorder(node.right)

    inorder(root)
    return self.res
```

### 297. Serialize and Deserialize Binary Tree

### 285. Inorder Successor in BST

Given a binary search tree and a node in it, find the in-order successor of that node in the BST.

The successor of a node p is the node with the smallest key greater than p.val.

https://blog.csdn.net/danspace1/article/details/86667504

#### Solution: Inorder Traversal O(n)

```python
def inorderSuccessor(self, root, p):
    def inOrder(root):
        if not root:
            return
        inOrder(root.left)
        l.append(root)
        inOrder(root.right)

    l = []
    inOrder(root)
    for i in range(len(l)):
        if l[i] == p:
            return l[i+1] if i+1 < len(l) else None
```

#### Solution: Binary O(h)

利用二叉搜索树的性质, 比p节点大的在它的右边, 我们只需找到比p节点值大的最小节点. 先将结果初始化为None, 如果p的值小于root, 那么我们更新结果为root, 然后往左边搜索, 当往右边搜索时, 我们不用更新结果, 因为要么之前已找到更小的结果, 要么还没找到比p节点的值更大的节点.

```python
def inorderSuccessor(self, root, p):
    ans = None
    while root:
        if p.val < root.val:
            ans = root
            root = root.left
        else:
            root = root.right
    return ans
```

### 270. Closest Binary Search Tree Value

Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.

Base on the characteristics of binary search tree to search for the target. O(logn)

https://goodtecher.com/leetcode-270-closest-binary-search-tree-value/

```python
def closestValue(self, root: TreeNode, target: float) -> int:
    self.closest = None
    self.search(root, target)
    return self.closest.val

def search(self, root, target):
    if not root:
        return

    if not self.closest or abs(root.val - target) < abs(self.closest.val - target):
        self.closest = root

    if target < root.val:
        self.search(root.left, target)        
    elif target > root.val:
        self.search(root.right, target)
```

### 272. Closest Binary Search Tree Value II

Given the root of a binary search tree, a target value, and an integer k, return the k values in the BST that are closest to the target. You may return the answer in any order.

You are guaranteed to have only one unique set of k values in the BST that are closest to the target.

https://www.goodtecher.com/leetcode-272-closest-binary-search-tree-value-ii/

https://github.com/leslierere/leetcode_with_python/blob/master/notes-tree.md

```python
def closestKValues(self, root: TreeNode, target: float, k: int) -> List[int]:
    if not root:
        return []

    nodel = []
    self.inorder(root, nodel)
    res = []
    i = 0

    # get the index in nodel that is just larger than the target
    while i < len(nodel):
        if nodel[i]>target:
            break
        i+=1

    while k>0:
        if i==0:
            res.append(nodel.pop(0))
        elif i== len(nodel):
            res.append(nodel.pop(-1))
            i-=1
        else:
            if abs(nodel[i-1]-target) < abs(nodel[i]-target):
                res.append(nodel.pop(i-1))
                i-=1
            else:
                res.append(nodel.pop(i))
        k -= 1

    return res

def inorder(self, node, nodel):
    if node:
        self.inorder(node.left, nodel)
        nodel.append(node.val)
        self.inorder(node.right, nodel)
```

一遍遍历，一边更新。值得好好看看

当size等于k的时候，还需要加入节点时我们就检查该节点和first element in the list 哪个离target更近，取近的留下。关于为什么跟最开始放进去的元素比较，我是认为因为前面那些元素因为res.size() < k, 所以会无脑放进去，但很有可能他们离target是很远的，所以我们要从first开始比较.

```python
def closestKValues(self, root: TreeNode, target: float, k: int) -> List[int]:
    res = deque()

    def ino(node):
        if node is None:
            return
        ino(node.left)
        if len(res) < k:
            res.append(node.val)
        elif abs(node.val - target) < abs(res[0] - target):
            res.popleft()
            res.append(node.val)
        else:
            return
        ino(node.right)

    ino(root)
    return res
```

### 99. Recover Binary Search Tree

题目交换了两个数字，其实就是在有序序列中交换了两个数字。而我们只需要把它还原。交换的位置的话就是两种情况。

1. 相邻的两个数字交换:

[ 1 2 3 4 5 ] 中 2 和 3 进行交换，[ 1 3 2 4 5 ]，这样的话只产生一组逆序的数字（正常情况是从小到大排序，交换后产生了从大到小），3 2。

我们只需要遍历数组，找到后，把这一组的两个数字进行交换即可。

2. 不相邻的两个数字交换:

[ 1 2 3 4 5 ] 中 2 和 5 进行交换，[ 1 5 3 4 2 ]，这样的话其实就是产生了两组逆序的数字对。5 3 和 4 2。

所以我们只需要遍历数组，然后找到这两组逆序对，然后把第一组前一个数字和第二组后一个数字进行交换即完成了还原。

```python
def recoverTree(self, root: Optional[TreeNode]) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    def ino(root):
        if root:
            ino(root.left, res)
            res.append(root)
            ino(root.right, res)

    res = []
    ino(root)

    first, second = None, None
    for i in range(len(res)-1):
        if res[i].val > res[i+1].val and not first:
            first = res[i]
        if res[i].val > res[i+1].val and first:
            second = res[i+1]

    first.val, second.val = second.val, first.val
```

### 116. Populating Next Right Pointers in Each Node

#### Solution: Level Order Traversal

取queue的最前面的作为next node，很妙

```python
def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None
        queue = [root]
        while queue:
            n = len(queue)
            for i in range(n):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if i == n - 1:
                    break
                node.next = queue[0]
        return root
```

#### Solution: Recursive

```python
def connect1(self, root):
    if root and root.left and root.right:
        root.left.next = root.right
        if root.next:
            root.right.next = root.next.left #这个是关键
        self.connect(root.left)
        self.connect(root.right)
```

### 117. Populating Next Right Pointers in Each Node II

传统level order可以参考上一题，但这样是O(n) space。 下面这种是O(1), 值得看看。

The algorithm is a BFS or level order traversal. We go through the tree level by level. node is the pointer in the parent level, tail is the tail pointer in the child level.
The parent level can be view as a singly linked list or queue, which we can traversal easily with a pointer.
Connect the tail with every one of the possible nodes in child level, update it only if the connected node is not nil.

```python
def connect(self, node):
    tail = dummy = TreeLinkNode(0)
    while node:
        tail.next = node.left#dummy.next = node.left
        if tail.next:#如果有左崽子
            tail = tail.next#tail变成左崽子
        tail.next = node.right#如果没有左崽子，dummy.next = node.right
        if tail.next:#如果有右崽子
            tail = tail.next#tail变成右崽子
        node = node.next#在本层移动
        if not node:
            tail = dummy
            node = dummy.next#崽子层最左边一个
```

### 314. Binary Tree Vertical Order Traversal

### 96. Unique Binary Search Trees

### 95. Unique Binary Search Trees II

