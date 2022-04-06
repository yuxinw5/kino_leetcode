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

