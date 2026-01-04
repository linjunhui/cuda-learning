# Python数据结构算法学习

## 学习目标

熟练掌握常用数据结构和算法，能够使用Python实现各种数据结构和算法，并分析算法复杂度。

## 学习时间

1个月（30天）

## 学习内容

### 第1周：Python内置数据结构深入理解
- 列表（list）深入
  - 列表内部实现原理
  - 列表的时间复杂度分析
  - 列表的扩展和优化
  - 列表的切片操作原理
- 字典（dict）深入
  - 字典的哈希表实现
  - 字典的时间复杂度分析
  - 字典的冲突解决
  - 字典的扩容机制
- 集合（set）深入
  - 集合的哈希表实现
  - 集合操作的时间复杂度
  - 集合的应用场景
- 元组（tuple）深入
  - 元组的不可变性
  - 元组的内存优化
  - 命名元组（namedtuple）应用

### 第2周：线性数据结构
- 数组和动态数组
  - 数组概念和实现
  - 动态数组实现
  - 数组操作的时间复杂度
- 链表
  - 单链表（Singly Linked List）
  - 双向链表（Doubly Linked List）
  - 循环链表（Circular Linked List）
  - 链表操作（插入、删除、查找）
  - 链表应用场景
- 栈（Stack）
  - 栈的概念和特性
  - 栈的实现（列表实现、链表实现）
  - 栈的应用（表达式求值、括号匹配、函数调用）
- 队列（Queue）
  - 队列的概念和特性
  - 普通队列实现
  - 双端队列（deque）
  - 优先队列（Priority Queue）
  - 队列的应用（BFS、任务调度）

### 第3周：树形数据结构
- 二叉树
  - 二叉树的概念和性质
  - 二叉树的遍历（前序、中序、后序、层序）
  - 二叉树的实现
  - 二叉搜索树（BST）
  - 平衡二叉树（AVL树）
- 堆（Heap）
  - 堆的概念和性质
  - 最小堆和最大堆
  - 堆的实现
  - 堆排序
  - 优先队列的堆实现
- 其他树结构
  - 红黑树概念
  - B树和B+树概念
  - 字典树（Trie）
  - 并查集（Union-Find）

### 第4周：图算法和高级算法
- 图的表示
  - 邻接矩阵
  - 邻接表
  - 图的实现
- 图遍历算法
  - 深度优先搜索（DFS）
  - 广度优先搜索（BFS）
- 最短路径算法
  - Dijkstra算法
  - Floyd-Warshall算法
- 最小生成树
  - Kruskal算法
  - Prim算法
- 排序算法
  - 冒泡排序、选择排序、插入排序
  - 快速排序、归并排序、堆排序
  - 计数排序、桶排序、基数排序
  - 排序算法复杂度分析
- 搜索算法
  - 线性搜索
  - 二分搜索
  - 哈希表搜索
- 动态规划
  - 动态规划概念
  - 状态转移方程
  - 经典问题（背包问题、最长公共子序列等）
- 贪心算法
  - 贪心算法概念
  - 贪心算法应用

## 实践项目

### 项目1：线性数据结构实现（第1-2周）
实现线性数据结构：
- 实现单链表
- 实现栈和队列
- 实现优先队列
- 测试和性能分析

### 项目2：树结构实现（第3周）
实现树形数据结构：
- 实现二叉搜索树
- 实现堆
- 实现树的遍历算法
- 测试和性能分析

### 项目3：算法实现和应用（第4周）
实现和应用算法：
- 实现图遍历算法
- 实现排序算法
- 实现动态规划问题
- LeetCode中等难度题目50道

## 学习资源

### 书籍
- 《算法导论》
- 《数据结构与算法分析》
- 《Python数据结构与算法分析》

### 在线资源
- LeetCode算法题库
- Python标准库文档（collections, heapq）
- VisuAlgo可视化算法

### 练习平台
- LeetCode（推荐）
- 牛客网
- HackerRank
- CodeForces

## 每日学习计划

### 工作日（2-3小时）
- 理论学习：1小时
- 编程实践：1-2小时

### 周末（4-6小时）
- 理论学习：2小时
- 编程实践：2-4小时

## 检查点

### 第1周检查点
- [ ] 深入理解Python内置数据结构的实现原理
- [ ] 能够分析数据结构的时间复杂度
- [ ] 掌握列表、字典、集合的内部机制

### 第2周检查点
- [ ] 能够实现链表、栈、队列
- [ ] 理解线性数据结构的应用场景
- [ ] 完成项目1

### 第3周检查点
- [ ] 能够实现二叉树和堆
- [ ] 掌握树的遍历算法
- [ ] 理解树结构的应用
- [ ] 完成项目2

### 第4周检查点
- [ ] 掌握图算法
- [ ] 实现常用排序和搜索算法
- [ ] 理解动态规划和贪心算法
- [ ] 完成项目3和50道LeetCode题目

## 代码示例

### 链表实现示例
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, val):
        if not self.head:
            self.head = ListNode(val)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = ListNode(val)
    
    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.val)
            current = current.next
        return elements
```

### 栈实现示例
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("栈为空")
        return self.items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("栈为空")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

### 二叉搜索树示例
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert(self.root, val)
    
    def _insert(self, root, val):
        if not root:
            return TreeNode(val)
        if val < root.val:
            root.left = self._insert(root.left, val)
        else:
            root.right = self._insert(root.right, val)
        return root
    
    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, root, result):
        if root:
            self._inorder(root.left, result)
            result.append(root.val)
            self._inorder(root.right, result)
```

## 常见问题

### Q: 什么时候使用列表，什么时候使用链表？
A: 列表适合随机访问频繁的场景，链表适合频繁插入删除的场景。Python的list实际上是动态数组，不是链表。

### Q: 如何选择合适的数据结构？
A: 根据操作的时间复杂度要求选择：
- 需要快速查找：字典或集合
- 需要有序：列表或排序的数据结构
- 需要LIFO：栈
- 需要FIFO：队列
- 需要层次关系：树

### Q: 动态规划的关键是什么？
A: 动态规划的关键是找到状态转移方程和最优子结构。通常使用记忆化递归或迭代方式实现。

---

**学习开始时间**：待定  
**预计完成时间**：待定

