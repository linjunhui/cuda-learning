# 题目描述
在一个增序的整数数组里找到两个数，使它们的和为给定值。已知有且只有一对解。
输入输出样例
输入是一个数组（numbers）和一个给定值（target）。输出是两个数的位置，从 1 开始计数。
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
在这个样例中，第一个数字（2）和第二个数字（7）的和等于给定值（9）

# 解题思路
## solution 1 hash table
1. 创建一个字典，将数组中的每一个数作为键，索引作为值。
2. 遍历数组，对于每一个数，判断 target - num 是否在字典中。
3. 如果在字典中，则返回当前索引和字典中该数对应的索引。

```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        std::map<int, int> value2idx;
        int numbers_size = numbers.size();

        for(int i = 0; i < numbers_size; i++) {
            value2idx[numbers[i]] = i;
        }
        
        for(int i = 0; i < numbers_size; i++) {
            int find = target - numbers[i];
            if(value2idx.count(find)) {
                return {i+1, value2idx[find]+1};
            }
        }
        return {};
    }
};
```

## solution 2 double pointers
1. define two pointers, one at the beginning of the array, one at the end.
2. compare the sum of the values at the two pointers.
3. If the sum is less than target, move the left pointer to the right.
4. If the sum is greater than target, move the right pointer to the left.
```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int left = 0, right = numbers.size() -1 ;
        int sum = 0;

        while(left <= right) {
            sum = numbers[left] + numbers[right];
            if(sum < target) {
                left++;
            } else if(sum > target) {
                right--;
            } else {
                return {left+1, right+1};
            }
        }

        return {};
    }
};
```