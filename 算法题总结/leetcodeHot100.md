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



#### [2. 两数相加](https://leetcode.cn/problems/add-two-numbers/)

```
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(-1), pre = dummyHead;
        //t表示进位
        int t = 0;
        while (l1 != null || l2 != null || t != 0) {
            if (l1 != null) {
                t += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                t += l2.val;
                l2 = l2.next;
            }
            pre.next = new ListNode(t % 10);
            pre = pre.next;
            t /= 10;
        }

        return dummyHead.next;
    }
}
```



#### [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

方法一：

滑动窗口经典做法

```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int index = 0, max = 0, start = 0;
        Set<Character> set = new HashSet<>();
        for(int i=0; i<s.length(); i++){
            while(set.contains(s.charAt(i))){
                set.remove(new Character(s.charAt(start)));
                start++;
            }
            set.add(s.charAt(i));
            max = Math.max(max, i-start+1);
        }
        return max;
    }
}
```

方法二：

用map记录位置

```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        // 记录字符上一次出现的位置
        int[] last = new int[128];
        for(int i = 0; i < 128; i++) {
            last[i] = -1;
        }
        int n = s.length();

        int res = 0;
        int start = 0; // 窗口开始位置
        for(int i = 0; i < n; i++) {
            int index = s.charAt(i);
            start = Math.max(start, last[index] + 1);
            res   = Math.max(res, i - start + 1);
            last[index] = i;
        }

        return res;
    }
}
```



#### [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

根据中位数的定义，当 m+n是奇数时，中位数是两个有序数组中的第 (m+n)/2+1个元素，当 m+n 是偶数时，中位数是两个有序数组中的第 (m+n)/2 个元素和第 (m+n)/2+1 个元素的平均值。

方法一：

两个指针分别定位两个数组，值小的指针先走，最后得到第k个元素；

方法二：

A和B两个数组，要找到第k个元素，可以比较A[k/2-1]和B[k/2-1]，两者前面分别由k/2-1个元素，如果A[k/2-1]小于B[k/2-1]，那么A[k/2-1]以及前面的元素都可以排除掉。在剩下数组中找第k-k/2个元素；

```
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int totalLen = nums1.length + nums2.length;
        if(totalLen%2 == 1){
            return getKthElement(nums1, nums2, totalLen/2+1);
        }else{
            return ((double)getKthElement(nums1, nums2, totalLen/2) + (double)getKthElement(nums1, nums2, totalLen/2+1))/2;
        }
    }

    //从两个数组中找第k个数
    private int getKthElement(int[] nums1, int[] nums2, int k) {
        int len1 = nums1.length, len2 = nums2.length;
        int index1 = 0, index2 = 0;
        while(true){
            //边界情况
            if(index1==len1) return nums2[index2+k-1];
            if(index2==len2) return nums1[index1+k-1];
            if(k==1) return Math.min(nums1[index1], nums2[index2]);

            //正常情况
            int half = k/2;
                //Math.min是防止这一次数组越界
            int newindex1 = Math.min(index1+half, len1) -1;
            int newindex2 = Math.min(index2+half, len2) -1;
            if(nums1[newindex1] <= nums2[newindex2]){
                k -= (newindex1-index1+1);
                index1 = newindex1 + 1;
            }else{
                k -= newindex2 - index2 + 1;
                index2 = newindex2 + 1;
            }
        }
    }
}
```

方法三：

划分数组法，没搞懂



#### [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

```
class Solution {
    public String longestPalindrome(String s) {
        if(s==null || s.length()<2) return s;
        
        int maxStart = 0, maxEnd = 0, maxLen = 1;
        int len = s.length();
        boolean[][] dp = new boolean[len][len];
        for(int r=1; r<len; r++){
            for(int l=0; l<r; l++){
            	//转移方程
                if(s.charAt(l)==s.charAt(r) && (r-l<=2||dp[l+1][r-1])){
                    dp[l][r] = true;
                    if(r-l+1>maxLen){
                        maxStart = l;
                        maxEnd = r;
                        maxLen = r-l+1;
                    }
                }
            }
        }
        return s.substring(maxStart, maxEnd+1);
    }
}
```

注意遍历的顺序；



#### [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

双指针指两端；每一次移动height较小的那一个，这样保证移动后可能出现盛水量大于当前的情况；

```
class Solution {
    public int maxArea(int[] height) {
        int l = 0, r = height.length-1, max = 0;      
        while(l<r){
            int curCap = Math.min(height[l], height[r]) * (r-l);
            max = Math.max(max, curCap);
            if(height[l]>height[r]) r--;
            else l++;
        }
        return max;
    }
}
```



#### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

双指针

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



#### [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

```
class Solution {
    private String[] ss = new String[]{"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if(digits.equals("")) return res;
        dfs(digits, 0, res, new char[digits.length()]);
        return res;
    }

    private void dfs(String digits, int dIndex, List<String> res, char[] cs){
        if(dIndex==digits.length()){
            res.add(String.valueOf(cs));
            return;
        }
        String cur = ss[digits.charAt(dIndex)-'2'];
        for(int i=0; i<cur.length(); i++){
            cs[dIndex] = cur.charAt(i);
            dfs(digits, dIndex+1, res, cs);
        }
    }


}
```

深度优先搜索；注意使用char[]而不是string，这样进行了优化



#### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

```
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if(head==null) return head;
        ListNode hh = new ListNode(), fa=hh, sl=hh;
        hh.next = head;
        for(int i=0; i<n; i++) sl=sl.next;
        while(sl.next!=null){
            fa=fa.next;
            sl=sl.next;
        }
        fa.next = fa.next.next;
        return hh.next;
    }
}
```

双指针；引入虚拟头结点，注意循环终止条件



#### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```
class Solution {
    public boolean isValid(String s) {
        Stack<Character>stack = new Stack<Character>();
        for(char c: s.toCharArray()){
            if(c=='(')stack.push(')');
            else if(c=='[')stack.push(']');
            else if(c=='{')stack.push('}');
            else if(stack.isEmpty()||c!=stack.pop())return false;
        }
        return stack.isEmpty();
    }
}
```

用栈，需要注意的是，对于左括号，放入右括号；



#### [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)

```
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode head = new ListNode(), cur = head;
        while(list1!=null && list2!=null){
            if(list1.val>list2.val){
                cur.next = list2;
                list2=list2.next;
            }else{
                cur.next = list1;
                list1=list1.next;
            }
            cur = cur.next;
        }
        if(list1!=null) cur.next = list1;
        if(list2!=null) cur.next = list2;
        return head.next;
    }
}
```

链表的题为了好处理特殊情况，一般都会加一个虚拟头结点；



#### [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)

```
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList<>();
        backTrace(ans, new char[2*n], 0, n, 0, 0);
        return ans;
    }

    private void backTrace(List<String> ans, char[] ele, int dep, int n, int left, int leftNotMatch){
        if(dep==2*n){
            ans.add(String.valueOf(ele));
            return;
        }

        if(left<n){
            ele[dep] = '(';
            backTrace(ans, ele, dep+1, n, left+1, leftNotMatch+1);
        }

        if(leftNotMatch!=0){
            ele[dep] = ')';
            backTrace(ans, ele, dep+1, n, left, leftNotMatch-1);
        }

    }
}
```

回溯法；根据left以及leftNotMatch判断当前括号应该取哪一种，使得产生的括号组合合法；





#### [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

1、分治法合并

```
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        return mergeList(lists, 0, lists.length-1);
    }

    public ListNode mergeList(ListNode[] lists, int l, int r){
        if(l==r) return lists[l];
        if(l>r) return null;
        int mid = (l+r)>>1;
        return mergeTwoLists(mergeList(lists, l, mid), mergeList(lists, mid+1, r)); 
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2){
        if(l1==l2) return l1;
        ListNode head = new ListNode(0), tail = head;
        while(l1!=null && l2!=null){
            if(l1.val<l2.val){
                tail.next = l1;
                tail = l1;
                l1 = l1.next;
            }else{
                tail.next = l2;
                tail = l2;
                l2 = l2.next;
            }
        }
        if(l1!=null) tail.next = l1;
        if(l2!=null) tail.next = l2;
        return head.next;
    }

}
```

2、优先队列法

维护一个优先队列，每次取值最小的结点，取完后，把取出的结点的下一个结点加进去

```
class Solution {

    class Status implements Comparable<Status> {
        int val;
        ListNode ptr;
        Status(ListNode ptr, int val) {
            this.ptr= ptr;
            this.val = val;
        }
        public int compareTo(Status status2) {
            return this.val - status2.val;
        }
    }

    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<Status> queue = new PriorityQueue<Status>();
        for(ListNode l : lists) if(l!=null) queue.offer(new Status(l, l.val));
        ListNode head = new ListNode(), tail = head;
        while(!queue.isEmpty()){
            Status s = queue.poll();
            tail.next = s.ptr;
            tail = tail.next;
            if(tail.next!=null) queue.offer(new Status(tail.next, tail.next.val));            
        }
        return head.next;
    }

    
}
```



#### [31. 下一个排列](https://leetcode-cn.com/problems/next-permutation/)

```
class Solution {
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[i] >= nums[j]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void reverse(int[] nums, int start) {
        int left = start, right = nums.length - 1;
        while (left < right) {
            swap(nums, left, right);
            left++;
            right--;
        }
    }
}
```

从后往前找第一个较小的字符x，再从后往前找第一个比x大的字符y，交换，然后反转当前y之后的字符；



#### [32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)

##### 方法一：

```
class Solution {
    public int longestValidParentheses(String s) {
        Deque<Integer> stack = new LinkedList<>();
        stack.addLast(-1);
        int max = 0;
        for(int i=0; i<s.length(); i++){
            if(s.charAt(i)=='('){
                stack.addLast(i);
            }else{
                stack.removeLast();
                if(stack.isEmpty()){
                    stack.addLast(i);
                }else{
                    max = Math.max(max, i-stack.peekLast());
                }
            }
        }
        return max;
    }
}
```

用栈，栈保存的是下标；

栈底始终存着一个右括号（表示截断）；

遇到左括号直接入栈，遇到右括号，就将栈顶出栈，表示和一个左括号匹配，并计算长度；

##### 方法二：

```
class Solution {
    public int longestValidParentheses(String s) {
        int maxans = 0;
        int[] dp = new int[s.length()];
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = Math.max(maxans, dp[i]);
            }
        }
        return maxans;
    }
}
```

动规，dp[i]表示以下标为i的字符为结尾的最长有效括号的长度；

则

'('的dp[i]必定为0；

')'的dp[i]且前一个为'('，那么dp[i] = dp[i-2]+2;

')'的dp[i]且前一个为')'，且第i-dp[i-1]-1个字符为'('，那么dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2;

[最长有效括号 - 最长有效括号 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/longest-valid-parentheses/solution/zui-chang-you-xiao-gua-hao-by-leetcode-solution/)



#### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

```
class Solution {
    public int search(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return -1;
        }
        if (n == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            // l-mid有序
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            // mid-r有序
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

二分：l-mid，mid-r必然有一个是有序的，



#### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

方法一：

二分查找，先找>=target的第一个，然后再找大于target的第一个；（都是找左边界）

方法二：

先找左边界，然后找右边界

```
public int binarySearch(int[] nums, int target, boolean lower) {
        int left = 0, right = nums.length - 1, ans = nums.length;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target || (lower && nums[mid] >= target)) {
                right = mid - 1;
                ans = mid;
            } else {
                left = mid + 1;
            }
        }
        return ans;
}
```

lower为true的时候，找左边界，nums[mid]=target的时候，right也要移到mid-1；

lower为false的时候，找有边界二，nums[mid] = target的时候，left也要移到mid+1；



#### [49. 字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

解法是做哈希，关键在于key的构造

方法一：用value排序后的String作为key；

方法二：value中，每个出现次数大于 0 的字母和出现次数按顺序拼接成字符串，这个字符串做为key；

方法三：每个字符对应一个质数，value中字符对应的质数*出现次数的和，作为key；



#### [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

```
class Solution {
    public String minWindow(String s, String t) {
        char[] cs = s.toCharArray(), ct = t.toCharArray();
        int[] count = new int[128];
        for(char c:ct) count[c]--;
        String res = "";
        for(int l=0,r=0,cnt=0; r<cs.length; r++){
            count[cs[r]]++;
            if(count[cs[r]]<=0) cnt++;
            //包含了t之后，移动左指针
            if(cnt==ct.length){
                while(count[cs[l]]>0){
                    count[cs[l]]--;
                    l++;
                }
                if(res.equals("")||res.length()>(r-l+1)) res = s.substring(l,r+1);
            }
        }
        return res;
    }
}
```

滑动窗口：

用count去记录缺少的；

移动左指针的时候，只能移动count[cs[l]]>0的，这说明即使移了还是包含t；

用cnt记录包含了t的几个字符；



#### [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

```
class Solution {
    /*“在一维数组中找第一个满足某种条件的数”的场景就是典型的单调栈应用场景。*/
    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        //向新数组中插入哨兵
        int[] newheights = new int[len + 2];
        System.arraycopy(heights,0,newheights,1,len);
        newheights[0] = 0;
        newheights[newheights.length-1] = 0;
        heights = newheights;
        len += 2;
        //单调栈
        int res = 0;
        Deque<Integer> stack = new ArrayDeque<>(len);
        stack.addLast(0);
        for (int i = 1; i < len; i++) {
            while(heights[i] < heights[stack.peekLast()]){
                int curHeight = heights[stack.pollLast()];
                int width = i - stack.peekLast() - 1;
                res = Math.max(res, curHeight*width);
            }
            stack.addLast(i);
        }
        return res;
    }
}
```

维护一个单调递增栈，当当前遍历到的数比栈顶的数小的时候，那么高度为栈顶的数的最大矩阵能够确定下来；

栈左边要加0，比如2，1。。。的情况，2在开头，已经能确定高度为2最大矩阵，因此开头需要加0；

右边也要为0，否则在递增序列的情况下，按照模板无法计算；



#### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

```
class Solution {
    public int maximalRectangle(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        if(m==0) return 0;
        int[][] left = new int[m+1][n];
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(matrix[i][j]=='1') left[i+1][j] = ((j==0)? 0:left[i+1][j-1])+1;
            }
        }

        int res = 0;
        for(int j=0; j<n; j++){
            Deque<Integer> stack = new LinkedList<>();
            stack.addLast(m+1);
            for(int i=m; i>=0; i--){
                while(stack.peekLast()!=m+1 && left[i][j]<left[stack.peekLast()][j]){
                    int t = stack.pollLast();
                    res = Math.max(res, left[t][j]*(stack.peekLast()-i-1));
                }
                stack.addLast(i);
            }
        }

        return res;
    }
}
```

遍历每个结点，确定结点在当前行的最大连续1，left[i] [j]；

然后对每一列从底向上计算“柱形最大面积”，也就是以当前结点为矩形的右下角结点；



#### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

```
class Solution {
    private int[] sum;

    public int numTrees(int n) {
        sum = new int[n+1];
        sum[0] = 1;
        process(sum, n);
        return sum[n];
    }

    public void process(int[] sum, int n){
        for(int i=1; i<=n; i++){
            if(sum[i-1]==0) process(sum, i-1);
            if(sum[n-i]==0) process(sum, n-i);
            sum[n] += sum[i-1]*sum[n-i];
        }
    }

}
```

n个结点的二叉搜索树个数相同（无论是1到n还是3到n+3）；

因此可以递归的时候记忆；



#### [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

```
class Solution {
    private TreeNode pre = null;
    
    public boolean isValidBST(TreeNode root) {
        if(root==null) return true;
        boolean left = isValidBST(root.left);
        if(left && (pre==null || pre.val < root.val)){
            pre = root;
        }else{
            return false;
        }
        boolean right = isValidBST(root.right);
        return right;
    }
}
```

中序遍历为有序，只需要多一个pre指针，判断前后大小；



#### [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

```
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return process(root.left, root.right);
    }

    public boolean process(TreeNode left, TreeNode right){
        if(left==null && right==null) return true;
        if((left==null&&right!=null) || (left!=null&&right==null) || left.val!=right.val) return false;
        return process(left.left, right.right) && process(left.right, right.left);
    }
}
```

递归



#### [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

用队列进行层序遍历



#### [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

```
class Solution {
    public int maxDepth(TreeNode root) {
        if(root==null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```



#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return doBuild(preorder, inorder, 0, 0, inorder.length-1);
    }

    public TreeNode doBuild(int[] preorder, int[] inorder, int root, int left, int right){
        if(left>right) return null;
        TreeNode n = new TreeNode(preorder[root]);
        int i;
        for(i=left; i<=right && inorder[i]!=preorder[root]; i++){}
        n.left = doBuild(preorder, inorder, root+1, left, i-1);
        n.right = doBuild(preorder, inorder, root+(i-left)+1, i+1, right);
        return n;
    }

}
```

递归

#### [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

方法一：

```
class Solution {
    public void flatten(TreeNode root) {
        while(root!=null){
            if(root.left==null){
                root = root.right;
            }else{
                TreeNode lr = root.left;
                while(lr.right!=null) lr = lr.right;
                lr.right = root.right;
                root.right = root.left;
                root.left = null;
                root = root.right;
            }
        }
    }
}
```

遍历root以及root的右子树，如果当前结点有左子树，那么就把当前结点的右子树，挂到当前结点的左子树的最右结点，然后当前结点的左子树，换到当前结点的右子树上；

方法二：

```
private TreeNode pre = null;

public void flatten(TreeNode root) {
    if (root == null)
        return;
    flatten(root.right);
    flatten(root.left);
    root.right = pre;
    root.left = null;
    pre = root;
}
```

先序遍历的时候，遍历到当前结点，就把当前结点挂到pre的右子树，但是这样不行，pre的右子树没有遍历，会丢失信息；

因此可以用右左中的逆先序遍历的方式，遍历到当前结点，就把当前结点的右子树置为pre（右子树已经遍历，不会存在信息丢失）

方法三：

```
public void flatten(TreeNode root) { 
    if (root == null){
        return;
    }
    Stack<TreeNode> s = new Stack<TreeNode>();
    s.push(root);
    TreeNode pre = null;
    while (!s.isEmpty()) {
        TreeNode temp = s.pop(); 
        if(pre!=null){
            pre.right = temp;
            pre.left = null;
        }
        if (temp.right != null){
            s.push(temp.right);
        }
        if (temp.left != null){
            s.push(temp.left);
        } 
        pre = temp;
    }
}
```

提前存储左右结点的先序遍历



#### [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

```
class Solution {
    private int max;
    public int maxPathSum(TreeNode root) {
        if(root==null) return 0;
        max = Integer.MIN_VALUE;
        dfs(root);
        return max;
    }
    
    // 返回经过root的单边分支最大和， 即Math.max(root, root+left, root+right)
    public int dfs(TreeNode root){
        if(root==null) return 0;
        
        // 左边分支最大值，左边分支如果为负数还不如不选择
        int leftMax = Math.max(0, dfs(root.left));
        
        // 右边分支最大值，右边分支如果为负数还不如不选择
        int rightMax = Math.max(0, dfs(root.right));
        
        // left->root->right 作为路径与已经计算过历史最大值做比较
        max = Math.max(max, root.val + leftMax + rightMax);
        
        // 返回经过root的单边最大分支给当前root的父节点计算使用
        return root.val + Math.max(leftMax, rightMax);
    }

}
```



#### [128. 最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

```
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        for(int num:nums) set.add(num);
        int max = 0;
        for(int num:nums){
            if(!set.contains(num-1)){
                int cNum = num;
                int cMax = 1;
                while(set.contains(cNum+1)){
                    cNum += 1;
                    cMax += 1;
                }
                max = Math.max(max, cMax);
            }
        }
        return max;
    }
}
```

用set存储数组所有数，

对于每一个数x，判断x-1，是否存在，不存在，就处理：判断x+1,x+2......是否存在；并更新



#### [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

```
class Solution {
    public int singleNumber(int[] nums) {
        int x = 0;
        for(int num:nums) x^=num;
        return x;
    }
}
```



#### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

方法一：

```
class Solution {
    private int[] mem;

    public boolean wordBreak(String s, List<String> wordDict) {
        mem = new int[s.length()];
        Set<String> set = new HashSet<>();
        for(String c:wordDict) set.add(c);
        return doBreak(s, 0, set);
    }

    private boolean doBreak(String s, int start, Set<String> wordDict){
        if(start==s.length()) return true;
        if(mem[start]!=0) return (mem[start]==1)? true : false;
        for(int i=start+1; i<=s.length(); i++){
            String c = s.substring(start, i);
            if(wordDict.contains(c)) {
                if(doBreak(s, i, wordDict)){
                    mem[start] = 1;
                    return true;
                }else{
                    mem[start] = -1;
                }
            }
        }
        return false;
    }

}
```

记忆化回溯，mem[i]表示以i开头的子串是否能够被break；

"leetcode"能否 break，可以拆分为：
"l"是否是单词表的单词、剩余子串能否 break。
"le"是否是单词表的单词、剩余子串能否 break。
"lee"...以此类推

方法二：

```
public class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}
```

dp[i]为前i个字符串能否被break；

转移方程：*dp*[*i*]=*dp*[*j*] && *contains*(*s*[*j*..*i*−1])



#### [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

```
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode low = head, fast = head;
        while(true){
            if(low==null) return false;
            if(fast==null || fast.next==null) return false;            
            low = low.next;
            fast = fast.next.next;
            if(low==fast) return true;
        }
    }
}
```

快指针一次走两步，慢指针一次走一步，一旦相遇，那么就存在环



#### [142. 环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

```
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head, low = head;
        while(fast!=null && fast.next!=null){
            fast = fast.next.next;
            low = low.next;
            if(fast==low){
                ListNode c1 = head, c2 = fast;
                while(c1!=c2){
                    c1 = c1.next;
                    c2 = c2.next;
                }
                return c1;
            }
        }
        return null;
    }
}
```

low每次走一步，fast每次走两步，low与fast在c2相遇，

c1从头结点出发，每次走一步，c2也每次走一步，

相遇的就是环的入口



#### [146. LRU 缓存](https://leetcode-cn.com/problems/lru-cache/)

```
class LRUCache {

    private class Node{
        int key, value;
        Node pre, next;
        public Node(){}
        public Node(int k, int v){key=k; value=v;}
    }

    private Map<Integer,Node> map;
    private int cap;
    private Node head, tail;

    public LRUCache(int capacity) {
        cap = capacity;
        map = new HashMap<>();
        head = new Node(); tail = new Node();
        head.next = tail; tail.pre = head;
    }
    
    public int get(int key) {
        if(!map.containsKey(key)) return -1;
        Node n = map.get(key);
        Node pre = n.pre, next = n.next;
        pre.next = next; next.pre = pre;
        n.next = tail; n.pre = tail.pre;
        tail.pre.next = n; tail.pre = n;
        return n.value;
    }
    
    public void put(int key, int value) {
        if(!map.containsKey(key)){
            if(map.size()>=cap){
                Node d = head.next;
                map.remove(d.key);
                head.next = d.next;
                d.next.pre = head;
            }
            Node n = new Node(key, value);
            map.put(key, n);
            n.next = tail; n.pre = tail.pre;
            tail.pre.next = n; tail.pre = n;
        }else{
            map.get(key).value = value;
            get(key);
        }
    }
}
```

双向链表（自己维护，并设置虚拟头尾结点）+map（map存的是<key，Node>）



#### [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

方法一：使用递归的归并排序

方法二：使用迭代的归并排序

```
class Solution {
    public ListNode sortList(ListNode head) {
        int intv = 1, len = 0;
        ListNode res = new ListNode();
        res.next = head;
        for(ListNode h=head; h!=null; h=h.next,len++){}

        while(intv<len){
            ListNode pre = res, h = res.next;
            while(h!=null){
                ListNode h1, h2;
                int c1 = 0, c2 = 0;
                //模块一
                for(h1=h; c1<intv && h!=null; h=h.next,c1++){}
                if(c1!=intv) break;
                //模块二
                for(h2=h; c2<intv && h!=null; h=h.next,c2++){}
                //合并
                while(c1>0 && c2>0){
                    if (h1.val < h2.val) {
                        pre.next = h1;
                        h1 = h1.next;
                        c1--;
                    } else {
                        pre.next = h2;
                        h2 = h2.next;
                        c2--;
                    }
                    pre = pre.next;
                }
                //合并剩下的
                pre.next = c1==0? h2:h1;
                // 更新pre指针的位置
                while (c1 > 0 || c2 > 0) {
                    pre = pre.next;
                    c1--;
                    c2--;
                }
                pre.next = h;
            }
            intv *= 2;
        }
        return res.next;
    }
}
```

pre指针的运用是关键



#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

```
class Solution {
    public int rob(int[] nums) {
        if(nums.length==1) return nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0]; dp[1] = Math.max(dp[0], nums[1]);
        for(int i=2; i<nums.length; i++){
            dp[i] = Math.max(nums[i]+dp[i-2], dp[i-1]);
        }
        return Math.max(dp[nums.length-2], dp[nums.length-1]);
    }
}
```

dp[i] = Math.max(nums[i]+dp[i-2], dp[i-1])，前一个表示偷当前，后一个表示不偷当前，考虑第i-1个，注意这里并不是说一定偷第i-1个；

注意dp[1] = Math.max(dp[0], nums[1])；



#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

方法一：

```
class Solution {

    private class UnionFind{
        private int count;
        private int[] parent;
        private int[] rank;

        public UnionFind(char[][] grid){
            int n = grid.length;
            int m = grid[0].length;
            parent = new int[n*m];
            rank = new int[n*m];
            for(int i=0; i<n; i++){
                for(int j=0; j<m; j++){
                    if (grid[i][j] == '1') {
                        parent[i * m + j] = i * m + j;
                        ++count;
                    }
                }
            }
        }

        public int find(int x){
            if(parent[x]!=x) parent[x] = find(parent[x]);
            return parent[x];
        }

        public void merge(int x, int y){
            int rootx = find(x);
            int rooty = find(y);
            if(rootx!=rooty){
                if (rank[rootx] > rank[rooty]) {
                    parent[rooty] = rootx;
                } else if (rank[rootx] < rank[rooty]) {
                    parent[rootx] = rooty;
                } else {
                    parent[rooty] = rootx;
                    rank[rootx] += 1;
                }
                --count;
            }
        }

        public int getCount(){
            return this.count;
        }

    }

    public int numIslands(char[][] grid) {
        UnionFind u = new UnionFind(grid);
        int m = grid[0].length;
        for(int i=0; i<grid.length; i++){
            for(int j=0; j<grid[0].length; j++){
                if(grid[i][j]=='1') {
                    if(i-1>=0&&grid[i-1][j]=='1') u.merge((i-1)*m+j, i*m+j);
                    if(j-1>=0&&grid[i][j-1]=='1') u.merge(i*m+j-1, i*m+j);
                }   
            }
        }
        return u.getCount();
    }
}
```

并查集



方法二：

```
class Solution {

    private void dfs(char[][] grid, int i, int j){
        if(i<0||j<0||i>=grid.length||j>=grid[0].length||grid[i][j]!='1') return;
        grid[i][j] = '0';
        dfs(grid, i+1, j);
        dfs(grid, i, j+1);
        dfs(grid, i-1, j);
        dfs(grid, i, j-1);
    }

    public int numIslands(char[][] grid) {
        int res = 0;
        for(int i=0; i<grid.length; i++){
            for(int j=0; j<grid[0].length; j++){
                if(grid[i][j]=='1') {
                    dfs(grid, i, j);
                    res += 1;
                }
            }
        }
        return res;
    }
}
```

对每个为1的结点深搜，并把对应位置0，注意四个方向都要搜；



#### [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head==null) return null;
        ListNode pre = null, cur = head, next = cur.next;
        while(cur!=null){
            cur.next = pre;
            pre = cur;
            cur = next;
            if(next!=null) next = next.next; 
        }
        return pre;
    }
}
```

三指针



#### [207. 课程表](https://leetcode-cn.com/problems/course-schedule/)

```
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
    	//以key为先修课的课程存入set中
        Map<Integer,Set<Integer>> map = new HashMap<>();
        //nums[i]表示i的先修课的个数
        int[] nums = new int[numCourses];
        for(int i=0; i<prerequisites.length; i++){
            if(map.containsKey(prerequisites[i][1])){
                Set<Integer> set = map.get(prerequisites[i][1]);
                set.add(prerequisites[i][0]);
            }else{
                Set<Integer> set = new HashSet<>();
                set.add(prerequisites[i][0]);
                map.put(prerequisites[i][1], set);
            }
            nums[prerequisites[i][0]]++;
        }
        int t = 0;
        while((t=getZero(nums,map))!=-1){
            nums[t] = -1;
            for(Integer i:map.get(t)){
                nums[i]--;
            }
            map.remove(t);
        }
        return map.isEmpty();
    }

    private int getZero(int[] nums, Map<Integer,Set<Integer>> map){
        for(int i=0; i<nums.length; i++){
            if(nums[i]==0&&map.containsKey(i)) return i;
        }
        return -1;
    }

}
```

拓扑排序；



#### [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

```
class Trie {
    public Trie[] childrens;
    public boolean isEnd;

    public Trie() {
        childrens = new Trie[26];
        isEnd = false;
    }
    
    public void insert(String word) {
        Trie n = this;
        for(char c:word.toCharArray()){
            if(n.childrens[c-'a']==null) n.childrens[c-'a'] = new Trie();
            n = n.childrens[c-'a']; 
        }
        n.isEnd = true;
    }
    
    public boolean search(String word) {
        Trie n = startsPrefix(word);
        return n!=null && n.isEnd;
    }
    
    public boolean startsWith(String prefix) {
        return startsPrefix(prefix)!=null;
    }

    private Trie startsPrefix(String prefix){
        Trie n = this;
        for(char c:prefix.toCharArray()){
            if(n.childrens[c-'a']==null) return null;
            n = n.childrens[c-'a'];
        }
        return n;
    }

}
```

前缀树



#### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

方法一：

```
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> q = new PriorityQueue<>();
        for(int n:nums) addQueue(q, n, k);
        return q.peek();
    }
    private void addQueue(PriorityQueue<Integer> q, int n, int size){
        if(q.size()>=size){
            if(q.peek()>=n) return;
            q.poll();
        }
        q.add(n);
    }
}
```

小根堆



方法二：

快排+分治



#### [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

```
class Solution {
    public int maximalSquare(char[][] matrix) {
        int n = matrix.length, m = matrix[0].length, maxEdge = 0;
        if(m==0 || n==0) return 0;
        int[][] dp = new int[n][m];
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                if(matrix[i][j]=='1'){
                    if(i==0||j==0){
                        dp[i][j] = 1;
                    }else{
                        dp[i][j] = Math.min(Math.min(dp[i-1][j-1], dp[i-1][j]), dp[i][j-1])+1;
                    }
                    maxEdge = Math.max(maxEdge, dp[i][j]);
                }
            }
        }
        return maxEdge*maxEdge;
    }
}
```

dp[i] [j]表示以（i，j）为右下角的最大正方形的边长；

dp[i] [j] = min（上面，左面，左上）+1；



#### [234. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

```
class Solution {
    public boolean isPalindrome(ListNode head) {
        ListNode firstHalfEnd = endOfFirstHalf(head);
        ListNode s2 = reverseList(firstHalfEnd.next), s1 = head;
        // 注意条件为s2为null，这样就不用考虑奇数偶数的情况
        while(s2!=null){
            if(s1.val!=s2.val) return false;
            s1 = s1.next;
            s2 = s2.next;
        }
        reverseList(firstHalfEnd.next);
        return true;
    }

    private ListNode reverseList(ListNode head){
        ListNode prev = null;
        ListNode cur = head;
        while(cur!=null){
            ListNode next = cur.next;
            cur.next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }

    private ListNode endOfFirstHalf(ListNode head){
        ListNode fast = head;
        ListNode slow = head;
        while(fast.next!=null && fast.next.next!=null){
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

把链表用快慢指针分为前后两部分，反转后面一部分，然后依次将前后两部分比较，最后再反转后面一部分还原；注意快慢指针如何找前后部分



#### [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null || root.val==q.val || root.val==p.val) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left!=null && right!=null) return root;
        if(left!=null) return left;
        if(right!=null) return right;
        return null;
    }
}
```



#### [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

```
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length, tmp = 1;
        int[] answer = new int[n];
        answer[0] = 1;
        for(int i=1; i<n; i++) answer[i] = answer[i-1]*nums[i-1];
        for(int i=n-2; i>=0; i--){
            tmp *= nums[i+1];
            answer[i] *= tmp;
        }
        return answer;
    }
}
```

nums中每一个元素作为横坐标，answer每一个元素作为纵坐标；



#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

```
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] res = new int[nums.length-k+1];
        Deque<Integer> q = new LinkedList<>();
        for(int r=0; r<nums.length; r++){
            if(r>=k && nums[r-k]==q.peekFirst()) q.removeFirst();
            while(!q.isEmpty() && q.peekLast()<nums[r]) q.removeLast();
            q.addLast(nums[r]);
            if(r>=k-1) res[r-k+1] = q.peekFirst();
        }
        return res;
    }
}
```

维护一个递减的双向队列



#### [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

```
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int r=0, l=matrix[0].length-1;
        while(r<matrix.length && l>=0){
            if(matrix[r][l]==target) return true;
            else if(matrix[r][l]<target) ++r;
            else if(matrix[r][l]>target) --l;
        }
        return false;
    }
}
```

从矩阵的右上角开始做二分查找；



#### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

```
class Solution {

    public int numSquares(int n) {
        int[] dp = new int[n+1];
        for(int i=1; i<=n; i++){
            dp[i] = i;
            for(int j=1; i-j*j>=0; j++){
                dp[i] = Math.min(dp[i], dp[i-j*j]+1);
            } 
        }
        return dp[n];
    }

}
```

dp；

转移方程为：对于所有j，i-j*j>=0,

dp[i] = Math.min(dp[i], dp[i-j*j]+1);



#### [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

```
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if(root==null) return "[]";
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        while(!q.isEmpty()){
            TreeNode n = q.poll();
            if(n==null) sb.append("null");
            else{
                sb.append(n.val);
                q.add(n.left);
                q.add(n.right);
            }
            sb.append(",");
        }
        return sb.substring(0, sb.length()-1)+"]";
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if(data=="[]") return null;
        String[] ss = data.substring(1, data.length()-1).split(",");
        Queue<TreeNode> q = new LinkedList<>();
        TreeNode root = new TreeNode(Integer.valueOf(ss[0]));
        q.add(root);
        int i = 1;
        while(!q.isEmpty()){
            TreeNode n = q.poll();
            if(!ss[i].equals("null")){
                n.left = new TreeNode(Integer.valueOf(ss[i]));
                q.add(n.left);
            }
            ++i;
            if(!ss[i].equals("null")){
                n.right = new TreeNode(Integer.valueOf(ss[i]));
                q.add(n.right);
            }
            ++i;
        }
        return root;
    }
}
```

序列化与反序列化都需要队列，反序列化还需要i定位当前元素；（注意序列化后这不是一个完全二叉树）



#### [301. 删除无效的括号](https://leetcode-cn.com/problems/remove-invalid-parentheses/)

方法一：

```
class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        Set<String> set = new HashSet<>();
        set.add(s);
        while(!set.isEmpty()){
            for(String cs:set){
                if(isValid(cs)) res.add(cs);
            }
            if(!res.isEmpty()) return res;
            Set<String> nset = new HashSet<>();
            for(String cs:set){
                for(int i=0; i<cs.length(); i++){
                    if(i>0 && cs.charAt(i-1)==cs.charAt(i)) continue;
                    nset.add(cs.substring(0,i)+cs.substring(i+1));
                }
            }
            set = nset;
        }
        return null;
    }

    private boolean isValid(String s){
        int l = 0;
        for(char c:s.toCharArray()){
            if(c=='(') ++l;
            else if(c==')'){
                if(l==0) return false;
                --l;
            } 
        }
        return l==0;
    }
}
```

首先判断是否合法，可以遍历字符串，维护一个x，遇到(，x加一，遇到)，如果为0，肯定不合法，否则x减一，最后x为0才合法；

运用BFS的方法，每层删每个位置的一个；



方法二：

DFS

[删除无效的括号 - 删除无效的括号 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/remove-invalid-parentheses/solution/shan-chu-wu-xiao-de-gua-hao-by-leetcode-9w8au/)



#### [309. 最佳买卖股票时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

```
class Solution {
    public int maxProfit(int[] prices) {
        //0买入，1保持卖出（两天前就卖出），2（今天卖出），3处于冷冻期
        int[][] dp = new int[prices.length][4];
        dp[0][0] -= prices[0];
        for(int i=1; i<prices.length; i++){
            dp[i][0] = Math.max(dp[i-1][0], Math.max(dp[i-1][1]-prices[i], dp[i-1][3]-prices[i]));
            dp[i][1] = Math.max(dp[i-1][3], dp[i-1][1]);
            dp[i][2] = dp[i-1][0] + prices[i];
            dp[i][3] = dp[i-1][2];
        }
        return Math.max(dp[prices.length-1][1], Math.max(dp[prices.length-1][2], dp[prices.length-1][3]));
    }
}
```

注意划分为4个状态



#### [312. 戳气球](https://leetcode.cn/problems/burst-balloons/)

```
class Solution {
    public int maxCoins(int[] nums) {
        int n = nums.length;
        int[] tmp = new int[n+2];
        tmp[0] = 1;
        tmp[n+1] = 1;
        for(int i=0; i<n; i++) tmp[i+1] = nums[i];
        int[][] dp = new int[n+2][n+2];
        for(int len=3; len<=n+2; len++){
            for(int i=0; i<=n+2-len; i++){
                for(int k=i+1; k<i+len-1; k++){
                    int left = dp[i][k];
                    int right = dp[k][i+len-1];
                    dp[i][i+len-1] = Math.max(dp[i][i+len-1], tmp[k]*tmp[i]*tmp[i+len-1]+left+right);
                }
            }
        }
        return dp[0][n+1];
    }
}
```

dp[i] [j] 表示在 （i,j）开区间中，能够获得的硬币的最大数量；获得dp[i] [j]需要枚举开区间中的 k；

小技巧：将nums两边加入1，省去了考虑边界的情况；



#### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

一维DP：

```
class Solution {
    public int coinChange(int[] coins, int amount) {
        // 自底向上的动态规划
        if(coins.length == 0){
            return -1;
        }

        // memo[n]的值： 表示的凑成总金额为n所需的最少的硬币个数
        int[] memo = new int[amount+1];
        memo[0] = 0;
        for(int i = 1; i <= amount;i++){
            int min = Integer.MAX_VALUE;
            for(int j = 0;j < coins.length;j++){
                if(i - coins[j] >= 0 && memo[i-coins[j]] < min){
                    min = memo[i-coins[j]] + 1;
                }
            }
            // memo[i] = (min == Integer.MAX_VALUE ? Integer.MAX_VALUE : min);
            memo[i] = min;
        }

        return memo[amount] == Integer.MAX_VALUE ? -1 : memo[amount];
    }
}
```

注意这里不能先遍历硬币再遍历容量（和 完全背包的组合问题 区分）



二维DP：

```
class Solution {
    public int coinChange(int[] coins, int amount) {
            int n = coins.length;
            int[][] dp = new int[n+1][amount+1];
            for(int i=1; i<=n; i++) dp[i][0] = 0;
            for(int j=1; j<=amount; j++) dp[0][j] = Integer.MAX_VALUE;

            for(int i=1; i<=n; i++){
                for(int j=1; j<=amount; j++){
                    dp[i][j] = dp[i-1][j];
                    if(j>=coins[i-1]&&dp[i][j-coins[i-1]]!=Integer.MAX_VALUE) dp[i][j] = Math.min(dp[i][j-coins[i-1]]+1, dp[i][j]);
                }
            }
            return dp[n][amount]==Integer.MAX_VALUE? -1 : dp[n][amount];
        }
}
```

二维DP的两个for顺序没有要求，更好理解；二维dp中，01背包和完全背包的唯一区别在于转移方程中的 右边是i-1 还是i；



#### [337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/)

方法一：

记忆化搜索

```
class Solution {
    Map<TreeNode, Integer> map = new HashMap<>();
    public int rob(TreeNode root) {
        if(root==null) return 0;
        Integer res;
        if((res=map.get(root)) != null) return res;
        int val1 = root.val, val2 = 0;
        //偷当前
        if(root.left!=null) val1 += rob(root.left.left)+rob(root.left.right);
        if(root.right!=null) val1 += rob(root.right.left)+rob(root.right.right);
        //不偷当前
        val2 += rob(root.left) + rob(root.right);
        //记忆化
        res = Math.max(val1, val2);
        map.put(root, res);
        return res;
    }
}
```

方法二：

树状DP

```
class Solution {
 
    public int rob(TreeNode root) {
        int[] res = doRob(root);
        return Math.max(res[0], res[1]);
    }

    public int[] doRob(TreeNode root){
        if(root==null) return new int[]{0,0};
        int[] left = doRob(root.left);
        int[] right = doRob(root.right);
        return new int[]{Math.max(left[0],left[1])+Math.max(right[0],right[1]), root.val+left[0]+right[0]};
    }

}
```

关键在于doRob返回值，0表示不偷的最大值，1表示偷的最大值



#### [338. 比特位计数](https://leetcode.cn/problems/counting-bits/)

```
class Solution {
    public int[] countBits(int n) {
        int[] res = new int[n+1];
        res[0] = 0;
        for(int i=1; i<=n; i++){
            if(i%2==0) res[i] = res[i/2];
            else res[i] = res[i-1] + 1;
        }
        return res;
    }
}
```

两个转移方程：

偶数：res[i] = res[i/2]；奇数：res[i] = res[i-1] + 1;



#### [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

方法一：

一趟遍历，统计次数Map，然后用容量为k的小根堆

方法二：

快排

```
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int num : nums) map.put(num, map.getOrDefault(num,0)+1);
        List<int[]> list = new ArrayList<>();
        for(Map.Entry<Integer, Integer> e : map.entrySet()) list.add(new int[]{e.getKey(), e.getValue()});
        return quickSort(list, 0, list.size()-1, k);
    }

    private int[] quickSort(List<int[]> list, int start, int end, int k){
        int[] tmp = list.get(start);
        int l = start, r = end;
        while(l<r){
            while(r>l && list.get(r)[1]<=tmp[1]) r--;
            Collections.swap(list, r, l);
            while(r>l && list.get(l)[1]>=tmp[1]) l++;
            Collections.swap(list, r, l);
        }
        if(l<k-1){
            return quickSort(list, l+1, end, k);
        }else if(l>k-1){
            return quickSort(list, start, l-1, k);
        }else{
            int[] res = new int[k];
            for(int i=0; i<k; i++) res[i] = list.get(i)[0];
            return res;
        }
    }
}
```

一趟遍历，统计次数Map，然后分段快排；

方法三：

桶排序

一趟遍历，统计次数Map；放到根据次数放到不同的桶中，然后从次数大的桶遍历到次数小的桶，放进res中；



#### [394. 字符串解码](https://leetcode.cn/problems/decode-string/)

```
class Solution {
    public String decodeString(String s) {
        LinkedList<String> stack = new LinkedList<>();
        int i = 0;
        while(i<s.length()){
            char c = s.charAt(i++);
            //遇到数字
            if(c>='0' && c<='9'){
                int n = c-'0';
                while((c=s.charAt(i))>='0' && c<='9'){
                    n = n*10 + c-'0';
                    i++;
                }
                stack.addLast(String.valueOf(n));
            //遇到[或者字母
            }else if(c!=']'){
                stack.addLast(String.valueOf(c));
            //遇到]
            }else{
                StringBuilder sb = new StringBuilder();
                while(!stack.peekLast().equals("[")) sb.append(stack.removeLast());
                stack.removeLast();
                int count = Integer.valueOf(stack.removeLast());
                sb = sb.reverse();
                String cs = sb.toString();
                while(--count>0) sb.append(cs);
                stack.addLast(sb.reverse().toString());
            }
        }
        StringBuilder res = new StringBuilder();
        while(!stack.isEmpty()) res.append(stack.removeLast());
        return res.reverse().toString();
    }
}
```

用栈处理，注意栈存的是String而不是字符，处理时需要反转



#### [399. 除法求值](https://leetcode.cn/problems/evaluate-division/)

方法一：BFS

```
class Solution {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String,List<Pair>> map = new HashMap<>();
        for(int i=0; i<equations.size(); i++){
            String from = equations.get(i).get(0), to = equations.get(i).get(1);
            if(!map.containsKey(from)) map.put(from, new ArrayList<Pair>());
            if(!map.containsKey(to)) map.put(to, new ArrayList<Pair>());
            map.get(from).add(new Pair(to, values[i]));
            map.get(to).add(new Pair(from, 1.0/values[i]));
        }

        double[] rev = new double[queries.size()];
        for(int i=0; i<queries.size(); i++){
            String f = queries.get(i).get(0), t = queries.get(i).get(1);
            double res = -1.0;
            if(map.containsKey(f)&&map.containsKey(t)){
                res = 1.0;
                if(!f.equals(t)){
                    res = 1.0;
                    Map<String,Double> fMap = new HashMap<>();
                    for(String key:map.keySet()) fMap.put(key, -1.0);
                    fMap.put(f, 1.0);
                    Queue<String> queue = new LinkedList<>();
                    queue.add(f);
                    while(!queue.isEmpty() && fMap.get(t)<0){
                        String cs = queue.poll();
                        for(Pair p : map.get(cs)){
                            if(fMap.get(p.to)<0){
                                fMap.put(p.to, fMap.get(cs)*p.val);
                                queue.add(p.to);
                            }
                        }
                    }
                    res = fMap.get(t);
                }
            }
            rev[i] = res;
        }

        return rev;
    }

    private class Pair{
        String to;
        double val;
        public Pair(){}
        public Pair(String _to, double _val){to=_to; val=_val;}
    }
}
```

关键是建立无向图，A到B的边的值是A/B的值；

然后对queries中的每一个元素处理，要求a/c，就从a开始BFS，注意除了维护队列外，还要维护一个map，表示a/x的值，只有当map中，a/x为-1才能够更新 并将 x放入队列中；



方法二：FLOYD

```
class Solution {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        //建立点与数字的映射
        int nvars = 0;
        Map<String, Integer> variables = new HashMap<String, Integer>();
        for (int i = 0; i < equations.size(); i++) {
            if (!variables.containsKey(equations.get(i).get(0))) {
                variables.put(equations.get(i).get(0), nvars++);
            }
            if (!variables.containsKey(equations.get(i).get(1))) {
                variables.put(equations.get(i).get(1), nvars++);
            }
        }
        //建图
        double[][] graph = new double[nvars][nvars];
        for (int i = 0; i < nvars; i++) Arrays.fill(graph[i], -1.0);
        for (int i = 0; i < equations.size(); i++) {
            int va = variables.get(equations.get(i).get(0)), vb = variables.get(equations.get(i).get(1));
            graph[va][vb] = values[i];
            graph[vb][va] = 1.0 / values[i];
        }
        //用Folyd算法找点与点间的最短路径
        for (int k = 0; k < nvars; k++) {
            for (int i = 0; i < nvars; i++) {
                for (int j = 0; j < nvars; j++) {
                	//关键！！注意与求最短路径不同
                    if (graph[i][k] > 0 && graph[k][j] > 0) {
                        graph[i][j] = graph[i][k] * graph[k][j];
                    }
                }
            }
        }
        //构建答案
        double[] rev = new double[queries.size()];
        for(int i=0; i<queries.size(); i++){
            String f = queries.get(i).get(0), t = queries.get(i).get(1);
            double res = -1.0;
            if(variables.containsKey(f) && variables.containsKey(t)){
                int fi = variables.get(f), ti = variables.get(t);
                res = graph[fi][ti];
            }
            rev[i] = res;
        }

        return rev;
    }
}
```



方法三：

[除法求值 - 除法求值 - 力扣（LeetCode）](https://leetcode.cn/problems/evaluate-division/solution/chu-fa-qiu-zhi-by-leetcode-solution-8nxb/)

merge与find的做法



#### [406. 根据身高重建队列](https://leetcode.cn/problems/queue-reconstruction-by-height/)

```
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        // 身高从大到小排（身高相同k小的站前面）
        Arrays.sort(people, (a, b) -> {
            if (a[0] == b[0]) return a[1] - b[1];
            return b[0] - a[0];
        });

        LinkedList<int[]> que = new LinkedList<>();

        for (int[] p : people) {
            que.add(p[1],p);
        }

        return que.toArray(new int[people.length][]);
    }
}
```

贪心：身高从大到小排（身高相同k小的站前面），这是因为，身高高的 插入了，后面无论如何插入都不会对它有影响；然后按照k插入到que中；



#### [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)

```
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for(int n : nums) sum+=n;
        if(sum%2!=0) return false;
        sum /= 2;
        int[][] dp = new int[nums.length+1][sum+1];
        for(int i=1; i<=nums.length; i++){
            for(int j=0; j<=sum; j++){
                dp[i][j] = dp[i-1][j];
                if(j>=nums[i-1]) dp[i][j] = Math.max(dp[i][j], dp[i-1][j-nums[i-1]]+nums[i-1]);
            }
        }
        return dp[nums.length][sum]==sum;
    }
}
```

01背包的变种



#### [437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/)

方法一：

```
class Solution {

    public int pathSum(TreeNode root, int targetSum) {
        if(root==null) return 0;
        int ret = rootSum(root,targetSum);
        ret += pathSum(root.left, targetSum);
        ret += pathSum(root.right, targetSum);
        return ret;
    }

    public int rootSum(TreeNode root, int targetSum){
        int ret = 0;
        if(root==null) return ret;
        if(root.val==targetSum) ret++;
        ret += rootSum(root.left, targetSum-root.val);
        ret += rootSum(root.right, targetSum-root.val);
        return ret;
    }
}
```

双重递归，第一层递归枚举每个结点作为开始结点，第二层递归从开始结点往下找；



方法二：

```
class Solution {
    public int pathSum(TreeNode root, int targetSum) {
        Map<Integer,Integer> prefix = new HashMap<>();
        prefix.put(0, 1);
        return count(root, 0, prefix, targetSum);
    }

    private int count(TreeNode root, int curr, Map<Integer,Integer> prefix, int targetSum){
        if(root==null) return 0;

        curr += root.val;
        int ret = prefix.getOrDefault(curr-targetSum, 0);
        prefix.put(curr, prefix.getOrDefault(curr,0)+1);
        ret += count(root.left, curr, prefix, targetSum);
        ret += count(root.right, curr, prefix, targetSum);
        prefix.put(curr, prefix.get(curr)-1);

        return ret;
    }
}
```

前缀和，prefix中存的是，从根到当前结点，前缀和为key的个数为value，curr表示从根到当前结点的和；

curr-value=targetSum时，表示从根到当前结点之间，存在某一个结点X，X到当前结点的和为targetSum



#### [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

```
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        Map<Character,Integer> map = new HashMap<>();
        Map<Character,Integer> window = new HashMap<>();
        int valid = p.length(), l = 0, r = 0;
        for(int i=0; i<p.length(); i++) map.put(p.charAt(i), map.getOrDefault(p.charAt(i), 0)+1);

        while(r<s.length()){
            char c = s.charAt(r);
            if(map.containsKey(c)){
                window.put(c, window.getOrDefault(c, 0)+1);
                if(window.get(c)<=map.get(c)) valid--;
            }
            while(valid==0){
                char lc = s.charAt(l);
                if(r-l+1==p.length()) res.add(l);
                if(map.containsKey(lc)){
                    window.put(lc, window.get(lc)-1);
                    if(window.get(lc)<map.get(lc)) valid++;
                }
                l++;
            }
            r++;
        }

        return res;
    }
}
```

滑动窗口算法总结：

[我写了一首诗，把滑动窗口算法变成了默写题 - 找到字符串中所有字母异位词 - 力扣（LeetCode）](https://leetcode.cn/problems/find-all-anagrams-in-a-string/solution/hua-dong-chuang-kou-tong-yong-si-xiang-jie-jue-zi-/)



#### [448. 找到所有数组中消失的数字](https://leetcode.cn/problems/find-all-numbers-disappeared-in-an-array/)

方法一：

```
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        for(int i=0; i<nums.length; i++){
            while(nums[i]!=i+1 && nums[i]!=nums[nums[i]-1]){
                swap(nums, i, nums[i]-1);  
            }
        }
        List<Integer> res = new ArrayList<>();
        for(int i=0; i<nums.length; i++){
            if(nums[i]!=i+1) res.add(i+1);
        }
        return res;
    }

    public void swap(int[] nums, int i, int j){
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }
}
```

原地交换



方法二：

遍历一次数组，遇到nums[i]，把nums[nums[i]-1]位置上的数置为负数；

再遍历一次，遇到nums[i]为正数，则i+1这个数没有出现过



#### [461. 汉明距离](https://leetcode.cn/problems/hamming-distance/)

方法一：

```
class Solution {
    public int hammingDistance(int x, int y) {
        int s = x ^ y, ret = 0;
        while (s != 0) {
            s &= s - 1;
            ret++;
        }
        return ret;
    }
}
```

方法二：

java中的bitcount

![JDK源码学习--JDK中Integer类的BitCount方法实现过程](https://www.likecs.com/default/index/img?u=aHR0cHM6Ly9waWFuc2hlbi5jb20vaW1hZ2VzLzYxOC9jYmI0YTU3ZWZkMTkxNWNiYmFjZWI0ZGUwODFiOWRiYS5wbmc=)

[JDK源码学习--JDK中Integer类的BitCount方法实现过程 - 爱码网 (likecs.com)](https://www.likecs.com/show-204334080.html)



#### [494. 目标和](https://leetcode.cn/problems/target-sum/)

```
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0, size = 0;
        for(int i=0; i<nums.length; i++) sum += nums[i];
        if(sum<Math.abs(target) || (target + sum)%2!=0) return 0;
        size = (target + sum)/2;
        int[][] dp = new int[nums.length+1][size+1];
        dp[0][0] = 1;
        for(int i=1; i<=nums.length; i++){
            for(int j=0; j<=size; j++){
                dp[i][j] = dp[i-1][j];
                if(j>=nums[i-1]) dp[i][j] += dp[i-1][j-nums[i-1]];
            }
        }
        return dp[nums.length][size];
    }
}
```

01背包；

left组合 - right组合 = target -> 

left - (sum - left) = target -> 

left = (target + sum)/2；

注意target + sum不为偶数时直接返回0，sum小于target绝对值也要直接返回0；



#### [538. 把二叉搜索树转换为累加树](https://leetcode.cn/problems/convert-bst-to-greater-tree/)

```
class Solution {
    private int pre = 0;

    public TreeNode convertBST(TreeNode root) {
        process(root);
        return root;
    }

    public void process(TreeNode root){
        if(root==null) return;
        process(root.right);
        root.val = root.val + pre;
        pre = root.val;
        process(root.left);
    }

}
```

右中左遍历，pre记录前驱；



#### [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

```
class Solution {
    private int res;
    public int diameterOfBinaryTree(TreeNode root) {
        res = 1;
        process(root);
        return res-1;
    }

    private int process(TreeNode n){
        if(n==null) return 0;   
        int l = process(n.left), r = process(n.right);
        res = Math.max(res, l+r+1);
        return Math.max(l,r)+1;
    }
}
```

后序遍历；

维护一个res表示最长路径的结点数；

每次先比较，Math.max(res, l+r+1)，这样就能把不经过当前结点的上层结点的情况考虑到；

递归函数process，返回的是当前结点左右分支中最大值；



#### [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/)

```
class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();
        int pre = 0, res = 0;
        map.put(pre, 1);
        for(int i=0; i<nums.length; i++){
            pre += nums[i];
            res += map.getOrDefault(pre-k, 0);//注意这一步不能和下一步调换，
            map.put(pre, map.getOrDefault(pre,0)+1);
        }
        return res;
    }
}
```

不能使用滑动窗口，因为右窗口右移，不一定增加，左窗口左移，不一定减少；

用前缀和，pre[i]表示前i个数的和，pre[j] - pre[i] = k的话，表示i和j夹着的几个数据和为k；



#### [581. 最短无序连续子数组](https://leetcode.cn/problems/shortest-unsorted-continuous-subarray/)

方法一：

数组A，A排序后为B；

然后从左到右扫，A中第一个与B不相等的为左边界；

从右到左扫，A中第一个与B不相等的为右边界；



方法二：

```
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length, max = Integer.MIN_VALUE, min = Integer.MAX_VALUE, left = -1, right = -1;
        for(int i=0; i<n; i++){
            if(nums[i]<max) right = i;
            else max = nums[i];
            if(nums[n-i-1]>min) left = n-i-1;
            else min = nums[n-i-1];
        }
        return right==-1? 0 : right-left+1;
    }
}
```

数组分为三段numsA，numsB，numsC，要确定numsB的左右边界，l和r；

min为numsB和numsC的最小，numsA中的每一个数都比min要小，l比min要大；

max为numsA和numsB的最大，numsC中的每一个数都比max要大，r比max要小；

枚举每一个位置；



#### [617. 合并二叉树](https://leetcode.cn/problems/merge-two-binary-trees/)

```
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if(root1==null) return root2;
        if(root2==null) return root1;
        TreeNode l = mergeTrees(root1.left, root2.left);
        TreeNode r = mergeTrees(root1.right, root2.right);
        root1.left = l;
        root1.right = r;
        root1.val = root1.val + root2.val;
        return root1;
    }
}
```

后序遍历



#### [621. 任务调度器](https://leetcode.cn/problems/task-scheduler/)

方法一：

```
class Solution {
    public int leastInterval(char[] tasks, int n) {
        int time = 0;
        int[] nextValid = new int[26], rests = new int[26];
        Arrays.fill(nextValid, 1);
        for(int i=0; i<tasks.length; i++) rests[tasks[i]-'A']++;
        
        for(int i=0; i<tasks.length; i++){
            time++;
            int min = Integer.MAX_VALUE;
            for(int j=0; j<26; j++) if(rests[j]>0) min = Math.min(min, nextValid[j]);
            time = Math.max(time, min);
            int best = -1;
            for(int j=0; j<26; j++){
                if(rests[j]>0 && nextValid[j]<=time && (best==-1 || rests[j]>rests[best])) best = j;
            }
            nextValid[best] = time + n + 1;
            rests[best]--;
        }
        return time;
    }


}
```

模拟；nextValid[i]表示下次能用字母i的时间，rests表示剩余多少个；

每次time++，然后和nextValid中的最小比较，取最大的那个；

再遍历rests，取nextValid[j]<=time，而且rests最大的那个；

方法二：

```
class Solution {
    public int leastInterval(char[] tasks, int n) {
        Map<Character, Integer> freq = new HashMap<Character, Integer>();
        // 最多的执行次数
        int maxExec = 0;
        for (char ch : tasks) {
            int exec = freq.getOrDefault(ch, 0) + 1;
            freq.put(ch, exec);
            maxExec = Math.max(maxExec, exec);
        }

        // 具有最多执行次数的任务数量
        int maxCount = 0;
        for (Map.Entry<Character, Integer> entry : freq.entrySet()) {
            if (maxExec == entry.getValue()) ++maxCount;
        }

        return Math.max((maxExec - 1) * (n + 1) + maxCount, tasks.length);
    }
}
```

桶思想；

[【任务调度器】C++ 桶子_配图理解 - 任务调度器 - 力扣（LeetCode）](https://leetcode.cn/problems/task-scheduler/solution/tong-zi-by-popopop/)



#### [647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/)

方法一：

```
class Solution {
    public int countSubstrings(String s) {
        int n = s.length(), ans = 0;
        for (int i = 0; i < 2 * n - 1; ++i) {
            int l = i / 2, r = i / 2 + i % 2;
            while (l >= 0 && r < n && s.charAt(l) == s.charAt(r)) {
                --l;
                ++r;
                ++ans;
            }
        }
        return ans;
    }
}
```

枚举每一个回文中心；

方法二：

```
class Solution {
    public int countSubstrings(String s) {
        int n = s.length(), res = 0;
        boolean[][] dp = new boolean[n][n];
        for(int j=0; j<n; j++){
            for(int i=0; i<=j; i++){
                if((j-i<2||dp[i+1][j-1]) && s.charAt(i)==s.charAt(j)) {
                    dp[i][j] = true;
                    res++;
                }
            }
        }
        return res;
    }
}
```

动规；注意遍历顺序

dp[i] [j]表示从i到j，是否是一个回文子串；

转移方程为当 s[i] == s[j] && (j - i < 2 || dp[i + 1] [j - 1]) 时，dp[i][j]=true，否则为false，j - i < 2；

j - i < 2表示只有一个字符时是回文串；两个字符时，如果相等，比如aa，也是回文串；

dp[i + 1] [j - 1]表示，ababa，去掉两边的a，后仍然是回文串时，ababa才是回文串；

方法三：

manacher算法，后面的计算利用前面的结果；

[(123条消息) 什么是Manacher(马拉车)算法-java代码实现_数据结构和算法的博客-CSDN博客_manacher](https://blog.csdn.net/abcdef314159/article/details/119204961)



#### [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/)

```
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        LinkedList<Integer> stack = new LinkedList<>();
        int[] res = new int[temperatures.length];
        for(int i=0; i<temperatures.length; i++){
            while(!stack.isEmpty() && temperatures[stack.peekLast()]<temperatures[i]){
                res[stack.peekLast()] = i-stack.peekLast();
                stack.removeLast();
            }
            stack.addLast(i);
        }
        return res;
    }
}
```

维护一个单调递减栈



#### [547. 省份数量](https://leetcode-cn.com/problems/number-of-provinces/)

并查集

[[Python/C++/Java\] 多图详解并查集 - 省份数量 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/number-of-provinces/solution/python-duo-tu-xiang-jie-bing-cha-ji-by-m-vjdr/)



#### [380. O(1) 时间插入、删除和获取随机元素](https://leetcode-cn.com/problems/insert-delete-getrandom-o1/)

- 动态数组存储元素值
- 哈希表存储存储值到索引的映射。

381题只需要将哈希表中的value改成Set<Integer>就可以了



#### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

方法一：

dp遍历每一个数，当前的值为前面满足要求的数的值+1的最大值

```
class Solution {
    public int lengthOfLIS(int[] nums) {
        if(nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        int res = 0;
        Arrays.fill(dp, 1);
        for(int i = 0; i < nums.length; i++) {
            for(int j = 0; j < i; j++) {
                if(nums[j] < nums[i]) dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}
```

方法二：

二分查找法；新建一个数组，然后第一个数先放进去，然后第二个数和第一个数比较，如果说大于第一个数，那么就接在他后面，如果小于第一个数，那么就用二分查找法替换。（第一个大于等于arr[i]的数，替换它(value[j] = arr[i])，因为此时以arr[i]为结尾的子序列比以原value[j]结尾的子序列更有“潜力”，因为更小所以有更多的可能得到成更长的子序列）

    private static void binarySearch(int[] arr) {
        // 长度加1是为了好理解，value[maxLength]即表示maxLength长度的子序列最后一位的值
        int[] value = new int[arr.length + 1];   
        // 初始化第一个数
        int maxLength = 1;
        value[1] = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > value[maxLength]) {
                // 大于目前最大长度的子序列的最后一位，给value[]后边续上
                maxLength++;
                value[maxLength] = arr[i];
            } else {
                // 小于目前最大长度的子序列的最后一位，查找前边部分第一个大于自身的位置
                // 更新它
                int t = find(value, maxLength, arr[i]);
                value[t] = arr[i];
            }
        }
        System.out.println(maxLength);
    }
     
    // 二分查找
    private static int find(int[] value, int maxindex, int i) {
        int l = 1, r = maxindex;
     
        while (l <= r) {
            int mid = (l + r) / 2;
     
            if (i > value[mid]) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
     
        return l;
    }

序列是78912345，前三个遍历完以后tail是789，这时候遍历到1，就得把1放到合适的位置，于是在tail二分查找1的位置，变成了189（如果序列在此时结束，因为res不变，所以依旧输出3），再遍历到2成为129，然后是123直到12345 

衍生题目：合唱队形[AcWing 482. 合唱队形 - AcWing](https://www.acwing.com/solution/content/3805/)

