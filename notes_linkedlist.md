## Linked List

## 基础

### 206. Reverse Linked List

#### Solution: Iterative

这个版本更实用，要牢牢记住，注意要track tail node

```python
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    pre = None
    while head:
        temp = head.next
        head.next = pre
        pre = head
        head = temp
    return pre
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

#### Solution: Mark as visited

```python
def hasCycle(self, head: Optional[ListNode]) -> bool:
    while head:
        if head.val == None:
            return True
        head.val = None
        head = head.next
    return False
```

#### Solution: Two Pointers

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

### 24. Swap Nodes in Pairs

用dummy node，画图！

```python
def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
    res = ListNode(next=head)
    pre = res

    # 必须有pre的下一个和下下个才能交换，否则说明已经交换结束了
    while pre.next and pre.next.next:
        cur = pre.next
        nn = pre.next.next
        pre.next = nn
        cur.next = nn.next
        nn.next = cur
        pre = pre.next.next
    return res.next
```

### 328. Odd Even Linked List

太绝了！记录even node list starts的位置是关键，而且并不需要update这个值，最后做链接就好了

```python
def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
        return head

    odd = head
    even = head.next
    eHead = even # We have to keep where the even-node list starts

    while even and even.next:
        odd.next = odd.next.next
        even.next = even.next.next
        odd = odd.next
        even = even.next

    odd.next = eHead # the odd pointer currently points at the last node of the odd-node list

    return head
```

### 92. Reverse Linked List II

The idea is simple and intuitive: find linkedlist [m, n], reverse it, then connect m with n+1, connect n with m-1

Use dummy to handle None cases

```python
def reverseBetween(self, head: Optional[ListNode], m: int, n: int) -> Optional[ListNode]:
    if m == n:
        return head

    dummyNode = ListNode(0)
    dummyNode.next = head
    pre = dummyNode

    for i in range(m - 1):
        pre = pre.next

    # reverse the [m, n] nodes
    curr = pre.next
    nxt = curr.next

    for i in range(n-m):
        tmp = nxt.next
        nxt.next = curr
        curr = nxt
        nxt = tmp

    pre.next.next = nxt
    pre.next = curr

    return dummyNode.next
```

### 237. Delete Node in a Linked List

copy value and then delete the next node

```python
def deleteNode(self, node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    node.val = node.next.val
    node.next = node.next.next
```

### 19. Remove Nth Node From End of List

```python
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    head_dummy = ListNode()
    head_dummy.next = head

    slow, fast = head_dummy, head_dummy
    for _ in range(n):
        fast = fast.next

    while fast.next:
        slow = slow.next
        fast = fast.next

    slow.next = slow.next.next
    return head_dummy.next
```

### 83. Remove Duplicates from Sorted List

注意移除了之后还可能有重复值，所以dummy不向后移动

```python
def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    dummy.next = head

    res = dummy

    while dummy.next and dummy.next.next:
        cur = dummy.next
        nxt = cur.next
        if cur.val == nxt.val:
            dummy.next = nxt
            continue
        dummy = dummy.next
    return res.next
```

### 203. Remove Linked List Elements

因为一直check的是pre的下一个element，所pre.next存在也是必要条件

```python
def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
    dummy = ListNode(0)
    dummy.next = head

    pre = dummy
    while pre and pre.next:
        cur = pre.next
        if cur.val == val:
            pre.next = cur.next
            continue
        pre = pre.next
    return dummy.next
```

### 82. Remove Duplicates from Sorted List II

注意【1，1，1】这种情况要全部移除

```python
def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0)
    dummy.next = head

    pre = dummy
    meet = False
    while pre and pre.next and pre.next.next:
        cur = pre.next
        nxt = cur.next
        if cur.val == nxt.val:
            meet = True
            cur = cur.next
            nxt = cur.next
            if nxt == None:
                pre.next = None
                continue
            pre.next = cur
            continue
        if meet == True:
            pre.next = nxt
            meet = False
            continue
        pre = pre.next
    return dummy.next
```

### 369. Plus One Linked List

### 2. Add Two Numbers

记得存carry number

```python
def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = cur = ListNode()
    carry = 0
    while l1 or l2:
        if l1 and not l2:
            carry, s = divmod(l1.val + carry, 10)
            dummy.next = ListNode(s)
            l1 = l1.next
        elif l2 and not l1:
            carry, s = divmod(l2.val + carry, 10)
            dummy.next = ListNode(s)
            l2 = l2.next
        elif l1 and l2:
            carry, s = divmod(l1.val + l2.val + carry, 10)
            dummy.next = ListNode(s)
            l1 = l1.next
            l2 = l2.next
        dummy = dummy.next
    if carry:
        dummy.next = ListNode(carry)
    return cur.next
```

### 160. Intersection of Two Linked Lists

#### Solution: Intuitive

算出两个list的长度差，让长的list先前进这个长度差，然后一起前进，发现相同的即返回

```python
def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    lena = 0
    t = headA
    while t:
        t = t.next
        lena +=1

    lenb = 0
    t = headB
    while t:
        t = t.next
        lenb +=1

    curA = headA
    curB = headB
    if lena > lenb:
        for i in range(lena-lenb):
            curA = curA.next
    else:
        for i in range(lenb-lena):
            curB = curB.next
    while curB != curA:
        curB = curB.next
        curA = curA.next
    return curA
```

#### Solution: Tricky

You can prove that: say A length = a + c, B length = b + c, after switching pointer, pointer A will move another b + c steps, pointer B will move a + c more steps, since a + c + b + c = b + c + a + c, it does not matter what value c is. Pointer A and B must meet after a + c + b (b + c + a) steps. If c == 0, they meet at NULL.

```python
def getIntersectionNode(self, headA, headB):
    if headA is None or headB is None:
        return None

    pa = headA # 2 pointers
    pb = headB

    while pa is not pb:
        # if either pointer hits the end, switch head and continue the second traversal, 
        # if not hit the end, just move on to next
        pa = headB if pa is None else pa.next
        pb = headA if pb is None else pb.next

    return pa
```

### 21. Merge Two Sorted Lists

可以最后再处理剩下的list，在循环里处理不够简洁

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

## 提高

### 234. Palindrome Linked List

To save space:

```python
def isPalindrome(self, head: Optional[ListNode]) -> bool:
    fast = slow = head
    # find the mid node
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    # reverse the second half
    node = None
    while slow:
        nxt = slow.next
        slow.next = node
        node = slow
        slow = nxt
    # compare the first and second half nodes
    while node: # while node and head:
        if node.val != head.val:
            return False
        node = node.next
        head = head.next
    return True
```

### 143. Reorder List

#### Solution: Optimited Two Pointer

O(n) time, O(1) space

```python
def reorderList(self, head: Optional[ListNode]) -> None:
    if not head:
        return

    # find the mid point
    slow = fast = head 
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # reverse the second half in-place
    pre, node = None, slow
    while node:
        pre, node.next, node = node, pre, node.next
    
    # Merge in-place; Note : the last node of "first" and "second" are the same
    first, second = head, pre
    while second.next:
        first.next, first = second, first.next
        second.next, second = first, second.next
    return 
```

#### Solution: Two Pointer

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

### 142. Linked List Cycle II

Fast: 快的指针走的步数，每次走两步

Slow：慢指针走的步数，每次走一步

L1: 起点到cycle entry的距离

L2: cycle entry到尾部的距离

Fast = 2 Slow

相遇时，fast比slow多走一圈，所以有fast - slow = l2, 所以当前slow = L2

如图，可以算出阴影部分长度，亦即相遇点到尾部长度为L1 + L2 - slow = L1

此时加一个指针，让新指针和slow同时走，这两个指针相遇时则为entry

```python
def detectCycle(self, head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None
    while head != slow:
        slow = slow.next
        head = head.next
    return head
```

### 148. Sort List

用merge sort，体现一个分治的思想。两个关键点：1.快慢指针找middle point的方法要烂熟于心 2. merge用dummy head

```python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        fast, slow = head.next, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        start = slow.next
        slow.next = None
        l, r = self.sortList(head), self.sortList(start)
        return self.merge(l, r)
    
    def merge(self, h1, h2):
        dummy = tail = ListNode(None)
        while h1 and h2:
            if h1.val < h2.val:
                tail.next, h1 = h1, h1.next
            else:
                tail.next, h2 = h2, h2.next
            tail = tail.next
    
        tail.next = h1 or h2
        return dummy.next
```

### 25. Reverse Nodes in k-Group

#### Solution: Recursive

其实很好想，注意base case是长度不够k或者k是0或1，剩下的情况就是reverse k然后衔接

```python
def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
    l, node = 0, head
    while node:
        l += 1
        node = node.next
    if k <= 1 or l < k:
        return head

    pre = None
    current = head
    for _ in range(k):
        temp = current.next
        current.next = pre
        pre = current
        current = temp
    head.next = self.reverseKGroup(current, k)
    return pre
```

### 61. Rotate List

注意k可能大于原本的list长度，所以算长度是必须的，不能直接上来就rotate

```python
def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head:
        return head
    lastElement = head
    length = 1

    while lastElement.next:
        lastElement = lastElement.next
        length += 1

    k = k % length
    lastElement.next = head
    tempNode = head
    for _ in range( length - k - 1 ):
        tempNode = tempNode.next

    head = tempNode.next
    tempNode.next = None

    return head
```

### 86. Partition List

好聪明！我们知道，快排中之所以用相对不好理解的双指针，就是为了减少空间复杂度，让我们想一下最直接的方法。new 两个数组，一个数组保存小于分区点的数，另一个数组保存大于等于分区点的数，然后把两个数组结合在一起就可以了。

> ```java
> 1 4 3 2 5 2  x = 3
> min = {1 2 2}
> max = {4 3 5}
> 接在一起
> ans = {1 2 2 4 3 5}
> ```

数组由于需要多浪费空间，而没有采取这种思路，但是链表就不一样了呀，它并不需要开辟新的空间，而只改变指针就可以了。

```python
def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
    h1 = l1 = ListNode(0)
    h2 = l2 = ListNode(0)
    while head:
        if head.val < x:
            l1.next = head
            l1 = l1.next
        else:
            l2.next = head
            l2 = l2.next
        head = head.next
    l2.next = None
    l1.next = h2.next
    return h1.next
```

### 23. Merge k Sorted Lists

#### Solution: Divide and Conquer

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

### 147. Insertion Sort List

基本逻辑就是先切断，然后pre记录insert的位置，pre的下一个就应该是正确的head的位置。时间复杂度O(n^2)

```python
def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(-5000)
    pre = dummy
    while head:
        node = dummy
        head_next = head.next
        head.next = None
        while node and head.val>node.val:
            pre = node
            node = node.next
        head.next = pre.next
        pre.next = head
        head = head_next

    return dummy.next
```
