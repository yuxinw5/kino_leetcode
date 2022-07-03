## Linked List

## 基础

### 206. Reverse Linked List

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
