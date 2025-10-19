class Solution {
    public:
        vector<int> twoSum(vector<int>& numbers, int target) {
            std::map<int, int> value2idx;
            int numbers_size = numbers.size();
            
            for(int i = 0; i < numbers_size; i++) {
                value2idx[numbers[i]] = i;
            }
            
            // 要遍历全部，因为可能 存在 许多相同的元素
            for(int i = 0; i < numbers_size; i++) {
                int find = target - numbers[i];
                if(value2idx.count(find)) {
                    return {i+1, value2idx[find]+1};
                }
            }
            return {};
        }
    };