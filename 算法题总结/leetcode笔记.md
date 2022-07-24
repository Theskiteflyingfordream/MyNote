一、数组专题：

主要方法：排序；双指针；建立 “值-数组位置” 的map；快慢指针；回溯



#### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

```
class Solution {

    public int[] twoSum(int[] nums, int target) {
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if(map.containsKey(target - nums[i])){
                return new int[]{i, map.get(target - nums[i])};
            }
            map.put(nums[i], i);
        }
        return new int[2];
    }

}
```

关键：建立 “值-数组位置” 的map，一次遍历数组，查看map中是否存在“target-当前值”；

启发：需要保留原始位置信息，且要取值，可以建立 “值-数组位置” 的map



#### [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

```
class Solution {

    public int maxArea(int[] height) {
        int left = 0, right = height.length-1, result = 0;
        while(left < right){
            result = (Math.min(height[left], height[right])*(right-left) > result)? Math.min(height[left], height[right])*(right-left):result ;
            if (height[right] < height[left]) {
                right --;
            }else{
                left ++;
            }
        }
        return result;
    }

}
```

关键：双指针分别指向两端，每一次移动较小的那一端（这一端不会再成为边界，因为固定这一端，另一端无论怎么移动，都不会超过当前，不用考虑，故不会再称为边界）



#### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
    	int len = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < len-2; i++) {
        	//剪枝处理,确定第一个数后：如果第一个数与前面相等，退出；如果其加上前面三个>0，break；如果其加上后面三个<0，退出
        	if(nums[i]+nums[i+1]+nums[i+2]>0) break;
            if(nums[i]+nums[len-1]+nums[len-2]<0) continue;
            	//与前面相等，则不考虑，因为会重复
            if(i!=0 && nums[i]==nums[i-1]) continue;
            //双指针遍历后续
            int left = i+1, right = len-1;
            while(left < right){
                if(nums[i]+nums[left]+nums[right]==0){
                    List<Integer> ele = new ArrayList<>();
                    ele.add(nums[i]);
                    ele.add(nums[left]);
                    ele.add(nums[right]);
                    result.add(ele);
                    //找到一组符合条件后后，注意继续移动双指针到“不重复元素”位置
                    right--;
                    left++;
                    while (left < right && nums[left] == nums[left - 1]) left++;
                    while (left < right && nums[right] == nums[right + 1]) right--;
                }else if(nums[i]+nums[left]+nums[right]>0){
                    right--;
                }else{
                    left++;
                }
            }
        }
        return result;
    }
}
```

关键：先排序，然后对于每一个元素：双指针（指向两端）遍历其后续的元素，找合适的两个。



#### [16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)

关键：双指针，方法类似15，只不过在使用双指针的过程中需要与“全局最小”进行比较



#### [18. 四数之和](https://leetcode-cn.com/problems/4sum/)

关键：双指针，确定双重循环确定前两个数，然后双指针确定剩下两个数。在确定第一个数后进行剪枝，确定第二个数后进行剪枝。



#### [26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

```
class Solution {
    public int removeDuplicates(int[] nums) {
        int t = 0;
        for (int i = 0; i < nums.length; i ++ ) {
            if (i == 0 || nums[i] != nums[i - 1]) nums[t ++ ] = nums[i];
        }
        return t;
    }
}
```

关键：快指针一趟遍历，慢指针存放结果



回溯对比题目（39和40主要区别在于传入的层数）；for以及经典做法+

#### [39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

第一种解法（每层递归用for遍历元素，注意for中初始值为当前元素位置）

```
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> ele = new ArrayList<>();
        backtrace(candidates, target, result, ele, 0, 0);
        return result;
    }

    private void backtrace(int[] candidates, int target, List<List<Integer>> result, List<Integer> ele, int total, int idx){
        if(total == target){
            result.add(new ArrayList<>(ele));
            return;
        }
        for(int i = idx; i < candidates.length; i++){
        	//剪枝
            if(total+candidates[i] <= target){
                ele.add(candidates[i]);
                //由于相同元素可以重复选择，故传入i而不是i+1
                backtrace(candidates, target, result, ele, total+candidates[i], i);
                ele.remove(new Integer(candidates[i]));
            }else{
                break;
            }
        }
    }

}
```

第二种解法（经典）

```
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> combine = new ArrayList<Integer>();
        dfs(candidates, target, ans, combine, 0);
        return ans;
    }

    public void dfs(int[] candidates, int target, List<List<Integer>> ans, List<Integer> combine, int idx) {
        if (idx == candidates.length) {
            return;
        }
        if (target == 0) {
            ans.add(new ArrayList<Integer>(combine));
            return;
        } 
        // 选择当前数 
        if (target - candidates[idx] >= 0) {
            combine.add(candidates[idx]);
            //关键：传入的参数仍为idx
            dfs(candidates, target - candidates[idx], ans, combine, idx);
            combine.remove(combine.size() - 1);
        }
        // 不选择当前数
        dfs(candidates, target, ans, combine, idx + 1);
    }
}
```



#### [40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

第一种解法（经典）

```
class Solution {

    /*回溯*/
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        backTrace(candidates.length, 0, candidates, target, ans, new ArrayList<Integer>(), 0);
        return ans;
    }

    private void backTrace(int len, int depth, int[] candidates, int target, List<List<Integer>> ans, List<Integer> list, int sum){
        if(sum==target){
            ans.add(new ArrayList<>(list));
            return;
        }else if(len==depth){
            return;
        }

        int nexDifferent = depth+1;
        while(nexDifferent<len && candidates[nexDifferent]==candidates[depth]) nexDifferent++;

        backTrace(len, nexDifferent, candidates, target, ans, list, sum);

        for(int i = depth; i < nexDifferent && sum<=target; i++){
            sum += candidates[depth];
            list.add(candidates[depth]);
            backTrace(len, nexDifferent, candidates, target, ans, list, sum);
        }
        while(list.remove(new Integer(candidates[depth]))){}
    }
}
```

关键：排序后能够对重复元素进行统一处理（重复元素都不选，选一个，选两个。。。）

第二种解法（每层递归用for遍历元素，注意for中初始值为当前元素位置）

```
public class Solution {

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        int len = candidates.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }
        Arrays.sort(candidates);
        Deque<Integer> path = new ArrayDeque<>(len);
        dfs(candidates, len, 0, target, path, res);
        return res;
    }

    private void dfs(int[] candidates, int len, int begin, int target, Deque<Integer> path, List<List<Integer>> res) {
        if (target == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = begin; i < len; i++) {
            // 剪枝：减去 candidates[i] 小于 0，减去后面的 candidates[i + 1]、candidates[i + 2] 肯定也小于 0，因此用 break
            if (target - candidates[i] < 0) {
                break;
            }
            // 同一层相同数值的结点，从第 2 个开始，候选数更少，结果一定发生重复，因此跳过，用 continue
            if (i > begin && candidates[i] == candidates[i - 1]) {
                continue;
            }
            path.addLast(candidates[i]);
            // 元素不可以重复使用，这里递归传递下去的是 i + 1 而不是 i
            dfs(candidates, len, i + 1, target - candidates[i], path, res);
            path.removeLast();
        }
    }
}
```



#### [41. 缺失的第一个正数](https://leetcode-cn.com/problems/first-missing-positive/)

```
class Solution {
     public int firstMissingPositive(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while(nums[i] <= nums.length && nums[i] > 0){
                if(nums[i] != i+1 && nums[i] != nums[nums[i] - 1]){
                    /*交换nums[i]和nums[nums[i] - 1]*/
                    int t = nums[i];
                    nums[i] = nums[nums[i] - 1];
                    nums[nums[i] - 1] = t;
                }
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if(nums[i] != i+1) return i + 1;
        }
        return nums.length + 1;
    }
}
```

原地哈希法：把数组作为一个哈希，f[i] = i+1，交换完后再遍历一遍即可。



#### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

方法一：暴力法

当前柱子能够装的雨水 = 左右两边较小的最高柱子 - 当前柱子高；

对每个柱子计算左右两边较小的最高柱子的高度，然后计算雨水累加

方法二：单调栈

```
class Solution {
    public int trap(int[] height) {
       int len = height.length, res = 0;
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < len; i++) {
            while(!stack.isEmpty() && height[stack.peekLast()] < height[i]) {
                int top = stack.pollLast();
                //左边界不能算
                if(stack.isEmpty()) break;
                int left = stack.peekLast();
                res += (Math.min(height[left],height[i])-height[top]) * (i-left-1);
            }
            stack.addLast(i);
        }
        return res;
    }
}
```

维护一个高度单调递减的栈，当出现高度大于栈顶元素时，就可以计算出栈顶元素的装水量



#### [48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

```
class Solution {
   private int matrixDimension;

    public void rotate(int[][] matrix) {
        this.matrixDimension = matrix.length;
        divideAndConquer(1,matrix);
    }

    public void divideAndConquer(int p,int[][] matrix){
        if(p > this.matrixDimension/2 + 1){
            return;
        }
        for (int i = p-1; i < matrix.length-p; i++) {
            int t1 = matrix[i][matrix.length-p];
            matrix[i][matrix.length-p]=matrix[p-1][i];
            int t2 = matrix[matrix.length-p][matrix.length-i-1];
            matrix[matrix.length-p][matrix.length-i-1] = t1;
            t1 = matrix[matrix.length-i-1][p-1];
            matrix[matrix.length-i-1][p-1] = t2;
            matrix[p-1][i] = t1;
        }
        divideAndConquer(p+1,matrix);
    }
}
```

分治法，对最外层旋转，再对最内层旋转；

注意递归参数与交换下标的关系



#### [53. 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

方法一：

动态规划，dp[i]表示包含当前数字的连续子数组的最大和。

方法二：

```
class Solution {
    public int maxSubArray(int[] nums) {
        return divideAndConquer(nums,0,nums.length-1);
    }

    public int divideAndConquer(int[] nums,int l,int k){
        if(l==k){
            return nums[l];
        }
        int center = (l+k)/2;
        int leftMax = divideAndConquer(nums,l,center);
        int rightMax = divideAndConquer(nums,center+1,k);
        int s1=nums[center],s2=nums[center],now=0;
        for (int left = center-1; left >=l ; left--) {
            now+=nums[left];
            s1 = now>s1? now:s1;
        }
        now = 0;
        for (int right = center+1; right <= k; right++) {
            now+=nums[right];
            s2 = now>s2? now:s2;
        }
        return Math.max(Math.max(leftMax,rightMax),s1+s2-nums[center]);
    }
}
```

分治法，问题划分为包含中间元素的中间部分 以及 左右两部分 的最大子数组



#### [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

```
class Solution {

    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        traceBack(1,matrix,result);
        return result;
    }

    public void traceBack(int n,int[][] matrix,List<Integer> result){
        if(Math.min(matrix.length,matrix[0].length)<2*n-1) return;
        int i= n - 1,j = n - 1;
        for (; j <= matrix[0].length-n; j++) result.add(matrix[i][j]);
        for (i=n,j=matrix[0].length-n; i <= matrix.length-n; i++) result.add(matrix[i][j]);
        if(matrix.length-n > n-1){
            for (j=matrix[0].length-n-1,i=matrix.length-n; j >= n-1; j--) result.add(matrix[i][j]);
        }
        if(matrix[0].length-n > n-1){
            for (i=matrix.length-n-1,j=n-1; i > n-1 ; i--) result.add(matrix[i][j]);
        }
        traceBack(n+1,matrix,result);
    }
}
```

分治法，从外向内，注意参数关系



#### [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

```
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (o1, o2) -> {
            if(o1[0]>o2[0]) return 1;
            if(o1[0]==o2[0]) return 0;
            return -1;
        });
        List<int[]> intermediateResult = new ArrayList<>();
        int i = 0;
        while(i < intervals.length){
            int j = i;
            int max = intervals[i][1];
            while(j<intervals.length && max>=intervals[j][0]) {
                max = Math.max(max,intervals[j][1]);
                ++j;
            }
            int[] a = new int[]{intervals[i][0],max};
            intermediateResult.add(a);
            i = j;
        }
        return intermediateResult.toArray(new int[intermediateResult.size()][]);
    }
}
```

按照区间第一个元素排序+双指针合并重叠的区间



#### [57. 插入区间](https://leetcode-cn.com/problems/insert-interval/)





二、数据结构

#### [547. 省份数量](https://leetcode-cn.com/problems/number-of-provinces/)

并查集

[[Python/C++/Java\] 多图详解并查集 - 省份数量 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/number-of-provinces/solution/python-duo-tu-xiang-jie-bing-cha-ji-by-m-vjdr/)



#### [380. O(1) 时间插入、删除和获取随机元素](https://leetcode-cn.com/problems/insert-delete-getrandom-o1/)

- 动态数组存储元素值
- 哈希表存储存储值到索引的映射。

381题只需要将哈希表中的value改成Set<Integer>就可以了

