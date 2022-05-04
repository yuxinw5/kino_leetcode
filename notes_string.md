## String

### 28. Implement strStr()

@22.4.6

#### Solution: Brutal force

O(n*m) time: n - length of haystack m - length of needle

#### Solution: KMP

See details on problem 214 in this note

### 14. Longest Common Prefix

@22.4.6

用长度最小的作为参考比较，we look at the other strings to make sure that they have the same character at the same index. If they do not, then the "common" prefix is terminated at that point because we have already checked the previous characters and did not terminate.

### 58. Length of Last Word

@22.4.6

take advantage of split() in Python, it automatically trims the string

### 387. First Unique Character in a String

@22.4.6

用Counter计数，loop时一旦找到count为1的情况立马return index因为我们要找first

### 383. Ransom Note

@22.4.6

可以利用Counter的减法，类似交集并集 O(m+n)

### 344. Reverse String

@22.4.6

直接s[left],s[right] = s[right],s[left]交换，不需要存temp value

### 151. Reverse Words in a String

@22.4.6

reverse whole string and reverse seperate word

### 186. Reverse Words in a String II

@22.4.6

Reverse Words in a String in place

### 205. Isomorphic Strings

@22.4.6

one to one mapping

### 293. Flip Game

@22.4.6

You are playing the following Flip Game with your friend: Given a string that contains only these two characters: + and -, you and your friend take turns to flip two consecutive "++" into "--". The game ends when a person can no longer make a move and therefore the other person will be the winner.

Write a function to compute all possible states of the string after one valid move.

For example, given s = "++++", after one move, it may become one of the following states:

["--++","+--+", "++--"]

If there is no valid move, return an empty list [].

```python
def generatePossibleNextMoves(self, currentState: str) -> List[str]:
    results = []
    for i in range(0, len(currentState) - 1):
        if currentState[i:i+2] == "++":
            flip = currentState[:i] + "--" + currentState[i+2:]
            results.append(flip)
    return results
```

###  294. Flip Game II

@22.4.6

You are playing the following Flip Game with your friend: Given a string that contains only these two characters: + and -, you and your friend take turns to flip two consecutive "++" into "--". The game ends when a person can no longer make a move and therefore the other person will be the winner.
Write a function to determine if the starting player can guarantee a win.

Example:
Input: s = "++++"
Output: true 

Explanation: The starting player can guarantee a win by flipping the middle "++" to become "+--+".

#### Solution: Brutal Force

如果我翻转了一次之后，生成的新的字符串朋友必输，那么就代表我必胜。怎么判断朋友输不输呢？递归调用自身即可。

```python
def canWin(self, s):
    for i in range(len(s) - 1):
        if s[i:i + 2] == "++" and not self.canWin(s[:i] + "--" + s[i + 2:]):
            return True
    return False
```

#### Solution: Brutal Force + Memoization

第一种思路比较慢，而且打印出来每次调用的s可以发现，会出现重复的s，所以此时不妨采取记忆化的方法，用一个哈希表存储每次调用的答案。

```python
def canWin(self, string):
    record = {}
    def helper(s):
        if s in record:
            return record[s]
        for i in range(len(s) - 1):
            if s[i:i + 2] == "++":
                next_s = s[:i] + "--" + s[i + 2:]
                if not helper(next_s):
                    record[next_s] = False
                    return True
        return False
    return helper(string)
```

#### Solution: Minimax

https://leetcode.com/problems/stone-game-ii/discuss/345222/python-minimax-dp-solution

没看明白，要再研究研究

### 290. Word Pattern

@22.4.6

one to one mapping, exactly the same as problem 205

### 242.Valid Anagram

@22.4.6

same as problem 383, except that we need to check whether the values in Counter are all zero.

### 49. Group Anagrams

@22.4.6

利用anagram中字符全部一致的特性，用sort做common feature。

key = ''.join(sorted(word))  // not sort string in place

### 249. Group Shifted Strings

@22.4.6

Given a string, we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd". We can keep "shifting" which forms the sequence:

"abc" -> "bcd" -> ... -> "xyz"

Given a list of strings which contains only lowercase alphabets, group all strings that belong to the same shifting sequence.

For example, given: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"], Return:

[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]

```python
def groupStrings(strings):
    dic = defaultdict(list)
    for ele in strings:
        #use immutable tuple as key
        pattern = ()
        for i in range(len(ele)-1):
            # mod handles negative difference
            diff = (ord(ele[i]) - ord(ele[i+1])) % 26
            pattern = pattern + (diff, )
        dic[pattern].append(ele)
    return list(dic.values())
```

### 87. Scramble String

@22.4.6

#### Solution: Intuitive Recursion

```python
def isScramble(self, s1: str, s2: str) -> bool:
    if s1 == s2:
        return True
    if sorted(s1) != sorted(s2):
        return False
    for i in range(1, len(s1)):
        # split but not swap
        if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
            return True
        # split and swap
        if self.isScramble(s1[:i], s2[-i:]) and self.isScramble(s1[i:], s2[:-i]):
            return True
    return False
```

#### Solution: Recursion + Memoization

```python
dic = {}
def isScramble(self, s1: str, s2: str) -> bool:
    if s1 == s2:
        return True
    if sorted(s1) != sorted(s2):
        self.dic[s1 + " " + s2] = False
        return False
    if s1 + " " + s2 in self.dic:
        return self.dic[s1 + " " + s2]

    for i in range(1,len(s1)):
        if (self.isScramble(s1[:i], s2[-i:]) and self.isScramble(s1[i:], s2[:-i])) or (self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:])):
            self.dic[s1 + " " + s2] = True
            return True

    self.dic[s1 + " " + s2] = False
    return False
```

#### Solution: DP

https://leetcode.wang/leetCode-87-Scramble-String.html

### 161. One Edit Distance

@22.4.6

Given two strings s and t, determine if they are both one edit distance apart.

There are 3 possiblities to satisify one edit distance apart:

    1. Insert a character into s to get t

    2. Delete a character from s to get t

    3. Replace a character of s to get t

Input: s = "ab", t = "acb"
Output: true

Explanation: We can insert 'c' into s to get t.

```python
def isOneEditDistance(self, s: str, t: str) -> bool:
     s_size = len(s)
     t_size = len(t)
     count = 0
     if s == t or abs(s_size - t_size)>1: return False
     if s_size == t_size:
         for i in range(s_size):
             if s[i]!=t[i]:
                 count+=1
             if count>1: return False
         return True
     elif s_size > t_size:
         for i in range(t_size):
             if t[i]!=s[i] and t[i]!=s[i+1]:
                 return False
     else:
         for i in range(s_size):
             if s[i]!=t[i] and s[i]!=t[i+1]:
                 return False
     return True
```
 
add or delete的情况转化为了比较连续两个字母是否一致，好棒
 
### 358. Rearrange String k Distance Apart

@22.4.6
 
Given a non-empty string str and an integer k, rearrange the string such that the same characters are at least distance k from each other.

All input strings are given in lowercase letters. If it is not possible to rearrange the string, return an empty string "".

* str = "aaadbbcc", k = 2

* Answer: "abacabcd"

* Another possible answer is: "abcabcda"

#### Solution: Priority Queue

使用Counter统计每个字符出现的次数，然后使用priority queue(python默认最小堆)，每次弹出出现次数最多的字符，添加到生成结果字符串的末尾。如果剩余的不同字符个数不够k，并且有一个字符需要放入两次，那么说明不能满足题目的要求，返回空字符串。另外，每次弹出出现次数最多的字符之后，不能直接放入堆中，因为直接放入堆中可能下次又被弹出来，所以应该放入一个临时的数组中，在单次操作结束之后再重新插入堆中。

```python
def rearrangeString(self, s: str, k: int) -> str:
    if k==0:
        return s
    counter = collections.Counter(s)
    pq = [(-counter[key], key) for key in counter] 
    heapq.heapify(pq)
    ans = ""
    while pq:
        temp = []
        if len(pq) < k and -pq[0][0] > 1:
            return ""
        for _ in range(min(k,len(pq))):
            item = heapq.heappop(pq)
            count = -item[0]
            c = item[1]
            
            ans += c
            count -= 1
            
            if count:
                temp.append((count,c))
                
        for count, c in temp:
            heapq.heappush(pq, (-count,c))
    return ans
```

https://www.youtube.com/watch?v=DT--N9p_O4Y

### 316. Remove Duplicate Letters

@22.4.6

#### Solution: Stack + greedy

```python
def removeDuplicateLetters(self, s: str) -> str:
    stack = []

    for i, c in enumerate(s):
        if c not in stack:
            while stack and stack[-1] > c and stack[-1] in s[i+1:]:
                stack.pop()
            stack.append(c)

    return ''.join(stack)
```

Explanation:

1. why we only consider it if the char not in stack:
   Between the entering char and the same char currently in the stack, if there is a lower order right after the char in stack, the char will not be in the stack at all, if the right-after one is higher order, pluck the already-in-stack-same char and use the entering char would only make the situation worse

2. when should we pluck out chars in the stack

   * clearly,  if we need pluck out one, there must be one after the current char
   * also the plucked one should have higher order, as we always want the lower orders to be at the front

3. should we pluck out all the higher order ones in the stack or as long as we moved downward in the stack and  reached one of no higher order we should stop

   * the answer is we would never confront with a situation to skip chars in stack and pluck out chars below them.

### 271. Encode and Decode Strings

@22.5.2

Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

#### Solution: 

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

### 168. Excel Sheet Column Title

@22.5.2

Now we can see that ABCD＝A×26³＋B×26²＋C×26¹＋D＝1×26³＋2×26²＋3×26¹＋4

But how to get the column title from the number? We can't simply use the n%26 method because:

ZZZZ＝Z×26³＋Z×26²＋Z×26¹＋Z＝26×26³＋26×26²＋26×26¹＋26

We can use (n-1)%26 instead, then we get a number range from 0 to 25.

### 171. Excel Sheet Column Number

@22.5.2

记得用reversed并且要记住position以便用指数

### 13. Roman to Integer & 12. Integer to Roman

@22.5.2

### 246. Strobogrammatic Number

@22.5.2

A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).

Example: "69"->True "88"->True

```python
def isStrobogrammatic(self, num):
    if len(num) == 0:
        return False

    for i in range((len(num) + 1) // 2):
        c1 = num[i]
        c2 = num[len(num) - 1 - i]
        if c1 == '8' == c2 == '8':
            pass
        elif c1 == '0' == c2:
            pass
        elif c1 == '1' == c2:
            pass
        elif (c1 == '6' and c2 == '9') or (c1 == '9' and c2 == '6'):
            pass
        else:
            return False

    return True
```

### 247. Strobogrammatic Number II

@22.5.2

A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).

Find all strobogrammatic numbers that are of length = n.

Example:
Input:  n = 2
Output: 
["11","69","88","96"]

```python
def strobogrammatic_num(n):
    return numdef(n, n)

def numdef(n, length):
      
    if n == 0: return [""]
    if n == 1: return ["1", "0", "8"]
      
    middles = numdef(n - 2, length)
    result = []
      
    for middle in middles:
        if n != length:            
            result.append("0" + middle + "0")
  
        result.append("8" + middle + "8")
        result.append("1" + middle + "1")
        result.append("9" + middle + "6")
        result.append("6" + middle + "9")
    return result
```

## Substring

### 76. Minimum Window Substring

@22.5.2

sliding window的关键在于什么时候扩张和什么时候缩紧，本题巧妙运用了两个Counter来跟踪当前窗口的字母和个数，并且用一个var来表示当前是否满足条件。满足
的条件是：1. s中的字符c存在于t中 2. c在t中的个数和c在当前窗口下的个数一致

#### Solution: Sliding Window
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

### 30. Substring with Concatenation of All Words

@22.5.2

#### Solution: Sort of straight forward- two maps

比较intuitive的解法，通过scan s来向后找满足条件的substring，只不过控制了i的范围和j的跳跃步数来缩短run time。巧妙的运用了seen这个dictionary来判断当前index是否符合条件。

```python
def findSubstring(self, s: str, words: List[str]) -> List[int]:
    wordBag = Counter(words)
    wordLen, numWords = len(words[0]), len(words)
    totalLen, res = wordLen*numWords, []
    for i in range(len(s)-totalLen+1):
        seen = defaultdict(int)
        for j in range(i, i+totalLen, wordLen):
            currWord = s[j:j+wordLen]
            if currWord in wordBag:
                seen[currWord] += 1
                if seen[currWord] > wordBag[currWord]:
                    break
            else:
                break
        if seen == wordBag:
            res.append(i)
    return res
```

### 3. Longest Substring Without Repeating Characters

@22.5.2

### Solution: 传统sliding window解法

一旦发现重复的，window是缩减到上一次出现这个字符的index下一位

```python
def lengthOfLongestSubstring(self, s: str) -> int:
    right = left = 0
    letters = set()
    max_len = 0

    while right < len(s):
        char = s[right]
        while char in letters:
            letters.remove(s[left])
            left += 1
        letters.add(char)
        max_len = max(max_len, right-left+1)
        right += 1

    return max_len
```

### Solution: sliding window + map

不需要另一个loop来缩进window，用map记录下字符出现的index

```python
def lengthOfLongestSubstring(self, s: str) -> int:
    if not s:
        return 0
    res = 1
    l = 0
    seen = {}
    for r in range(len(s)):
        if s[r] in s[l:r]:
            l = seen[s[r]] + 1
        seen[s[r]] = r
        res = max(res,r-l+1)
    return res
```

### 340. Longest Substring with At Most K Distinct Characters

@22.5.2

Given a string, find the length of the longest substring T that contains at most k distinct characters.

For example, Given s = “eceba” and k = 2, T is "ece" which its length is 3.

跟上一题一样，不过这次用不到map因为需要从left一步一步缩进，直到满足条件

```python
def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
    l = r = 0
    charDict = collections.defaultdict(int)
    res = []

    while r<len(s):
        charDict[s[r]]+=1
        while len(charDict)>k:
            res.append(r-l)
            charDict[s[l]]-=1
            if charDict[s[l]]==0:
                charDict.pop(s[l])
            l+=1
        r+=1
    res.append(r-l)

    return max(res)
```

### 395. Longest Substring with At Least K Repeating Characters

@22.5.2

#### Solution： Sliding Window O(N)

进阶版sliding window：Imagine, that we have s = aabbb... and k = 3, what should we do when we reached window aabbb: should we expand it to the right hoping that we will meet another a? Or should we start to move left side of our window? One way to handle this problem is to do several sliding windows passes, where we fix T number of different chars we must have in our substring.

For each character group size eg 1 2 3 .. no of unique chars, we try to make group with t chars and freq of each char >=k within that group.

```python
def longestSubstring(str: str, k: int):
    total_uniques = len(Counter(str))
    ans = 0 

    for no_uniques in range(1, total_uniques+1):
        s = 0
        e = 0
        curr_sub = Counter()
        while e < len(str):
            curr_sub[str[e]] += 1 # first add the element at right index to the sub
            
            while len(curr_sub) > no_uniques: # if the number of unique characters in current sub exceedes no_uniques
                curr_sub[str[s]] -= 1 # removes elements from the left side
                if curr_sub[str[s]] == 0:
                    del curr_sub[str[s]] 
                s += 1   
                
            for key in curr_sub:
                if curr_sub[key] < k:
                    break
            else:
                ans = max(ans, e - s + 1)
            e += 1
    return ans
```

#### Solution: Divide and Conquer

