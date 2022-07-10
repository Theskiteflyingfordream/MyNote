#### [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

```
class CQueue {
    
    private Deque<Integer> stackHead;
    private Deque<Integer> stackTail;

    public CQueue() {
        stackHead = new ArrayDeque<Integer>();
        stackTail = new ArrayDeque<Integer>();

    }
    
    public void appendTail(int value) {
        stackTail.addLast(value);
    }
    
    public int deleteHead() {
        if(stackTail.isEmpty() && stackHead.isEmpty()){
            return -1;
        }else if(!stackHead.isEmpty()){
            return stackHead.removeLast();
        }else{
            while(!stackTail.isEmpty()){
                stackHead.addLast(stackTail.removeLast());
            }
            return stackHead.removeLast();
        }
    }
}
```

栈1用作push的栈，栈2用作pop；pop的时候，栈2如果非空，则直接弹，否则从栈1出，栈2入，再弹



#### [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

```
class MinStack {

    private Deque<Integer> stackA;
    private Deque<Integer> stackB;

    /** initialize your data structure here. */
    public MinStack() {
        stackA = new ArrayDeque<>();
        stackB = new ArrayDeque<>();
    }
    
    public void push(int x) {
        stackA.addLast(x);
        Integer t = stackB.peekLast();
        if((t!=null&&x<=t.intValue()) || t==null){
            stackB.addLast(x);
        }
    }
    
    public void pop() {
        if(!stackA.isEmpty()){
            if(stackA.removeLast().equals(stackB.peekLast())){
                stackB.removeLast();
            }
        }
    }
    
    public int top() {
       if(!stackA.isEmpty()){
           return stackA.peekLast();
       }else{
           return -1;
       }
    }
    
    public int min() {
        if(!stackB.isEmpty()){
            return stackB.peekLast();
        }else{
            return 0;
        }
    }
}
```

用一个辅助栈作为单调最小栈，弹出的时候，需要对应删除



#### [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

```
class Solution {
    public static int[] reversePrint(ListNode head) {
        ListNode node = head;
        int count = 0;
        while (node != null) {
            ++count;
            node = node.next;
        }
        int[] nums = new int[count];
        node = head;
        for (int i = count - 1; i >= 0; --i) {
            nums[i] = node.val;
            node = node.next;
        }
        return nums;
    }
}
```

用栈；或者两次遍历链表，第一次确定数组大小，第二次反向填充数组。

#### [剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

```
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head==null || head.next==null) return head;
        ListNode pre = null, nex = head.next;
        while(head!=null) {
            head.next = pre;
            pre = head;
            head = nex;
            if(nex!=null) nex = nex.next;
        }
        return pre;
    }
}
```

三指针



#### [剑指 Offer 35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

方法一：

```
class Solution {
    public Node copyRandomList(Node head) {
        if(head==null) return head;
        Map<Node, Node> map = new HashMap<>();
        for(Node cur = head; cur!=null; cur = cur.next){
            map.put(cur, new Node(cur.val));
        }
        for(Node cur = head; cur!=null; cur = cur.next){
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
        }
        return map.get(head);
    }
}
```

两次扫描链表，第一次用map将原node与复制node建立联系，第二次处理复制node的指针域

方法二：

```
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) {
            return head;
        }
        //将拷贝节点放到原节点后面，例如1->2->3这样的链表就变成了这样1->1'->2->2'->3->3'
        for (Node node = head, copy = null; node != null; node = node.next.next) {
            copy = new Node(node.val);
            copy.next = node.next;
            node.next = copy;
        }
        //把拷贝节点的random指针安排上
        for (Node node = head; node != null; node = node.next.next) {
            if (node.random != null) {
                node.next.random = node.random.next;
            }
        }
        //分离拷贝节点和原节点，变成1->2->3和1'->2'->3'两个链表，后者就是答案
        Node newHead = head.next;
        for (Node node = head, temp = null; node != null && node.next != null;) {
            temp = node.next;
            node.next = temp.next;
            node = temp;
        }

        return newHead;
    }
}
```

原地修改，不用额外空间



#### [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

```
//使用一个新的对象，复制 str，复制的过程对其判断，是空格则替换，否则直接复制，类似于数组复制
public static String replaceSpace(StringBuffer str) {
        if (str == null) {
            return null;
        }
		//选用 StringBuilder 单线程使用，比较快，选不选都行
        StringBuilder sb = new StringBuilder();
		//使用 sb 逐个复制 str ，碰到空格则替换，否则直接复制
        for (int i = 0; i < str.length(); i++) {
		//str.charAt(i) 为 char 类型，为了比较需要将其转为和 " " 相同的字符串类型
            if (" ".equals(String.valueOf(str.charAt(i)))){
                sb.append("%20");
            } else {
                sb.append(str.charAt(i));
            }
        }
        return sb.toString();
    }
```

很多数组填充类的问题，都可以先预先给数组扩容带填充后的大小，然后在从后向前进行操作。



#### [剑指 Offer 58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

```
class Solution {
    public String reverseLeftWords(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for(int i = n; i < s.length(); i++){
            sb.append(s.charAt(i));
        }
        for(int i = 0; i < n; i++){
            sb.append(s.charAt(i));
        }
        return sb.toString();
    }
}
```

如果要求不能使用额外空间，则可以通过局部反转+整体反转 达到左旋转的目的。

具体步骤为：

1. 反转区间为前n的子串
2. 反转区间为n到末尾的子串
3. 反转整个字符串

最后就可以得到左旋n的目的，而不用定义新的字符串，完全在本串上操作。

```
class Solution {
    public String reverseLeftWords(String s, int n) {
        int len=s.length();
        StringBuilder sb=new StringBuilder(s);
        reverseString(sb,0,n-1);
        reverseString(sb,n,len-1);
        return sb.reverse().toString();
    }
     public void reverseString(StringBuilder sb, int start, int end) {
        while (start < end) {
            char temp = sb.charAt(start);
            sb.setCharAt(start, sb.charAt(end));
            sb.setCharAt(end, temp);
            start++;
            end--;
            }
        }
}
```



#### [剑指 Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

```
class Solution { 

    //方法1：Set(或用boolean数组)，时间O(n)，空间O(n)，不修改原数据
     public int findRepeatNumber(int[] nums) {
         HashSet<Integer> set = new HashSet<>();
         for(int num:nums){
             if(set.contains(num)) return num;
             set.add(num);
         }
         return -1;
     }

    // //方法2：原地置换，时间O(n)，空间O(1)
    public int findRepeatNumber(int[] nums) {
        if(nums==null || nums.length==0) return -1;
        for(int i = 0 ; i < nums.length;i++){
            //如果该数字没有不和他的索引相等
            while(nums[i]!=i){
                //重复返回
                if(nums[i]==nums[nums[i]]){
                    return nums[i];
                }
                //不重复交换
                int temp = nums[nums[i]];
                nums[nums[i]] = nums[i];
                nums[i] = temp;
            }
        }
        return -1;
    }

    // //方法3：对0到n-1进行二分查找，时间O(nlogn)，空间O(1)，不修改原数据，用时间换空间
    //该方法需要数字一定有重复的才行，因此如果题目修改在长度为n，数字在1到n-1的情况下，此时数组中至少有一个数字是重复的，该方法可以通过。
     public int findRepeatNumber(int[] nums) {
         //统计nums中元素位于0到m的数量，如果数量大于这个值，那么重复的元素肯定是位于0到m的
         int min = 0 ;
         int max = nums.length-1;
         while(min<max){
             int mid = (max+min)>>1;
             int count = countRange(nums,min,mid);
             if(count > mid-min+1) {
                 max = mid;
             }else{
                 min = mid+1;
             }
         }
         最后min=max
         return min;
     }

     //统计范围内元素数量,时间O(n)
     private int countRange(int[] nums,int min,int max){
         int count = 0 ;
         for(int num:nums){
             if(num>=min && num<=max){
                 count++;
             }
         }
         return count;
     }
    

}
```



#### [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

```
class Solution {
    public int search(int[] nums, int target) {
        return helper(nums, target) - helper(nums, target - 1);
    }
    int helper(int[] nums, int tar) {
        int i = 0, j = nums.length - 1;
        while(i <= j) {
            int m = (i + j) / 2;
            if(nums[m] <= tar) i = m + 1;
            else j = m - 1;
        }
        return i;
    }
}
```

helper() 函数旨在查找数字 tartar 在数组 numsnums 中的 插入点 ，且若数组中存在值相同的元素，则插入到这些元素的右边。



#### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

```
class Solution {
    public int missingNumber(int[] nums) {
        int i = 0, j = nums.length - 1;
        while(i <= j) {
            int m = (i + j) / 2;
            if(nums[m] == m) i = m + 1;
            else j = m - 1;
        }
        // 跳出时，变量 i 和 j 分别指向 “右子数组的首位元素” 和 “左子数组的末位元素” 。因此返回 i 即可。
        return i;
    }
}
```

排序数组中的搜索问题，首先想到 二分法 解决。
根据题意，数组可以按照以下规则划分为两部分。
左子数组： nums[i] = i；
右子数组： nums[i] != i；
缺失的数字等于 “右子数组的首位元素” 对应的索引；因此考虑使用二分法查找 “右子数组的首位元素”



#### [剑指 Offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

```
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if(matrix.length==0) return false;
        int row = 0, line = matrix[0].length-1;
        while(row<matrix.length && line>=0){
            if(matrix[row][line]>target){
                line--;
            }else if(matrix[row][line]<target){
                row++;
            }else{
                return true;
            }
        }
        return false;
    }
}
```

根据右上角的元素性质，下面比它大，左边比它小，可以得出类似二分查找方法。



#### [剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

```
class Solution {
    public int minArray(int[] numbers) {
        int i = 0, j = numbers.length - 1;
        while (i < j) {
            int m = (i + j) / 2;
            if (numbers[m] > numbers[j]) i = m + 1;
            else if (numbers[m] < numbers[j]) j = m;
            else {
                int x = i;
                for(int k = i + 1; k < j; k++) {
                    if(numbers[k] < numbers[x]) x = k;
                }
                return numbers[x];
            }
        }
        return numbers[i];
    }
}
```

利用二分查找寻找旋转点，m作为i，j的中点：

当 nums[m] > nums[j] 时： m 一定在 右排序数组 中，即旋转点 x 一定在 [m + 1, j] 闭区间内；
当 nums[m] < nums[j] 时： m 一定在 左排序数组 中，即旋转点 x 一定在[i, m] 闭区间内；
当 nums[m] = nums[j] 时： 无法判断 m 在哪个排序数组中，即无法判断旋转点 x 在 [i, m] 还是 [m + 1, j] 区间中。解决方案： 执行 j = j - 1或者放弃二分用线性。



#### [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

方法一：

```

class Solution {
    public char firstUniqChar(String s) {
        HashMap<Character, Boolean> dic = new HashMap<>();
        char[] sc = s.toCharArray();
        for(char c : sc)
            dic.put(c, !dic.containsKey(c));
        for(char c : sc)
            if(dic.get(c)) return c;
        return ' ';
    }
}
```

第一次遍历字符数组构建哈希表，第二次遍历字符数组得到第一个value为true的字符。

方法二：

```
class Solution {
    public char firstUniqChar(String s) {
        Map<Character, Boolean> dic = new LinkedHashMap<>();
        char[] sc = s.toCharArray();
        for(char c : sc)
            dic.put(c, !dic.containsKey(c));
        for(Map.Entry<Character, Boolean> d : dic.entrySet()){
           if(d.getValue()) return d.getKey();
        }
        return ' ';
    }
}
```

在方法一的基础上使用LinkedHashMap，使键值对顺序与放入的顺序一致，第二次遍历时遍历键值对（这使得遍历次数减少）

方法三：

```
public char firstUniqChar(String s) {
        int[] arr = new int[26];
        char[] chars = s.toCharArray();
        for (char ch : chars){
            arr[ch -'a'] ++;
        }
        for (char c:chars){
            if (arr[c-'a'] == 1){
                return c;
            }
        }
        return ' ';
    }
```

使用数组代替哈希表



#### [面试题32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

```
class Solution {
    public int[] levelOrder(TreeNode root) {
        if(root==null) return new int [0];
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        ArrayList<Integer> ans = new ArrayList<>();
        while(!queue.isEmpty()){
            TreeNode cur = queue.poll();
            ans.add(cur.val);
            if(cur.left!=null) queue.add(cur.left);
            if(cur.right!=null) queue.add(cur.right);
        }
        int[] res = new int[ans.size()];
        for(int i = 0; i < res.length; i++){
            res[i] = ans.get(i);
        }
        return res;
    }
}
```

BFS，使用队列



#### [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

```
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        if(root==null) return new ArrayList<List<Integer>>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        List<List<Integer>> res = new ArrayList<>();
        while(!queue.isEmpty()){
            List<Integer> ele = new ArrayList<>();
            int size = queue.size();
            for(int i = 0; i < size; i++){
                TreeNode cur = queue.poll();
                ele.add(cur.val);
                if(cur.left!=null) queue.add(cur.left);
                if(cur.right!=null) queue.add(cur.right);
            }
            res.add(ele);
        }
        return res;
    }
}
```

类似上一条提



#### [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

```
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        if(root==null) return new ArrayList<List<Integer>>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        List<List<Integer>> res = new ArrayList<>();
        int prj = 1;
        while(!queue.isEmpty()){
            Deque<Integer> ele = new ArrayDeque<>();
            int size = queue.size();
            for(int i = 0; i < size; i++){
                TreeNode cur = queue.poll();
                if(prj==1){
                    ele.addLast(cur.val);
                }else{
                    ele.addFirst(cur.val);
                }
                if(cur.left!=null) queue.add(cur.left);
                if(cur.right!=null) queue.add(cur.right);
            }
            prj = ~prj;
            res.add(new ArrayList<Integer>(ele));
        }
        return res;
    }
}
```

BFS+双端队列+奇偶判断



#### [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

```
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        return (A != null && B != null) && (recur(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B));
    }
    boolean recur(TreeNode A, TreeNode B) {
        if(B == null) return true;
        if(A == null || A.val != B.val) return false;
        return recur(A.left, B.left) && recur(A.right, B.right);
    }
}
```

递归，B是A的子结构，则”A或A的左子树或A的右子树“ 中 出现和B相同的结构和节点值。



#### [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

方法一：递归，镜像右子树，作为当前结点的左子树，镜像左子树，作为当前结点的右子树

```
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if(root==null) return null;
        TreeNode tmp = root.left;
        root.left = mirrorTree(root.right);
        root.right = mirrorTree(tmp);
        return root;
    }
}
```

方法二：利用队列，交换每个结点的左右子树。



#### [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

```
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root==null) return true;
        return recursion(root.left, root.right);
    }

    public boolean recursion(TreeNode left, TreeNode right){
        if(left==null && right==null) return true;
        if((left==null&&right!=null) || (left!=null&&right==null) || left.val!=right.val) return false;
        return recursion(left.left,right.right) && recursion(left.right, right.left);
    }
}
```

做递归思考三步：

1. 递归的函数要干什么？

- 函数的作用是判断传入的两个树是否镜像。
- 输入：TreeNode left, TreeNode right
- 输出：是：true，不是：false

2. 递归停止的条件是什么？

- 左节点和右节点都为空 -> 倒底了都长得一样 ->true
- 左节点为空的时候右节点不为空，或反之 -> 长得不一样-> false
- 左右节点值不相等 -> 长得不一样 -> false

3.  从某层到下一层的关系是什么？

- 要想两棵树镜像，那么一棵树左边的左边要和二棵树右边的右边镜像，一棵树左边的右边要和二棵树右边的左边镜像
- 调用递归函数传入左左和右右
- 调用递归函数传入左右和右左
- 只有左左和右右镜像且左右和右左镜像的时候，我们才能说这两棵树是镜像的



#### [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

```
class Solution {
    public int fib(int n) {
        if(n<2){
            return n;
        }
        int p = 0, q = 1, r = 1;
        for(int i = 2; i < n; i++){
            p = q;
            q = r;
            r = (p+q)%1000000007;
        }
        return r;
    }
}
```

动态规划+滚动数组



#### [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

```
class Solution {
    public int numWays(int n) {
        if(n==0||n==1) return 1;
        int p = 1, q = 1;
        for(int i = 2; i <= n; i++){
            int sum = (p + q)%1000000007;
            p = q;
            q = sum;
        }
        return q;
    }
}
```

跳上n级台阶的跳法 = 跳上n-1级台阶的跳法 + 跳上n-2级台阶的跳法



#### [剑指 Offer 63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

```
class Solution {
    public int maxProfit(int[] prices) {
        int minCos = Integer.MAX_VALUE, profit = 0;
        for(int i = 0; i < prices.length; i++){
            minCos = Math.min(prices[i], minCos);
            profit = Math.max(profit, prices[i]-minCos);
        }
        return profit;
    }
}
```

dp[i]表示前i天卖出所得利润，则

​	转移方程：dp[i] = max(dp[i-1], dp[i] - minCos)

​	初始条件：dp[0] = 0

​	所求为dp[prices.length]

空间优化：由于dp[i]只与dp[i-1]有关，因此一维数组可以优化到一个变量。



#### [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

```
class Solution {
    public int maxSubArray(int[] nums) {
        int cur = nums[0];
        int maxSum = cur;
        for(int i = 1; i < nums.length; i++){
            cur = Math.max(cur+nums[i], nums[i]);
            maxSum = Math.max(maxSum, cur);
        }
        return maxSum;
    }
}
```

dp[i]表示包含第i个数字作为最后数字的连续子序列最大值，则

dp[i] = max(dp[i-1]+nums[i], nums[i]);

所有dp[i]中最大即为结果

空间优化：由于dp[i]只与dp[i-1]有关，因此一维数组可以优化到一个变量；同时，用一个变量记录当前最大。



#### [剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

```
class Solution {
    public int maxValue(int[][] grid) {
        int rows = grid.length, lines = grid[0].length;
        int[] dp = new int[lines];
        dp[0] = grid[0][0];
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < lines; j++){
                if(i==0&&j==0) continue;
                if(i-1>=0 && j-1>=0){
                    dp[j] = Math.max(dp[j-1]+grid[i][j], dp[j]+grid[i][j]);
                }else if(i-1>=0){
                    dp[j] = dp[j]+grid[i][j];
                }else if(j-1>=0){
                    dp[j] = dp[j-1]+grid[i][j];
                }
            }
        }
        return dp[lines-1];
    }
}
```

dp[i] [j]表示走到当前位置最多能拿到的价值，则dp[i] [j] = max(上一个位置+grid[i] [j], 左边位置+grid[i] [j])，对边界上边界和左边界特殊处理，返回dp[grid.length] [grid[0].length]则为结果。

滚动数组进行空间优化：当前行的计算之和上一行有关，因此只用一个一维数组存储结果即可。



#### [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

优化前：

```
class Solution {
    public int translateNum(int num) {
        if(num<10) return 1;
        char[] c = String.valueOf(num).toCharArray();
        int[] dp = new int[c.length];
        dp[0] = 1;
        dp[1] = ((c[0]-'0')*10+c[1]-'0'<26)? 2:1;  
        for(int i = 2; i < c.length; i++){
            dp[i] = ((c[i-1]!='0')&&((c[i-1]-'0')*10+c[i]-'0'<26))? dp[i-1]+dp[i-2]:dp[i-1];
        }
        return dp[c.length-1];
    }
}
```

dp[n]表示当前字符串的前n个字符构成的字符串的翻译方法，则

​	dp[n] = dp[n-1]+dp[n-2]，当第n-1个字符非0 且 与当前字符构成数小于26

​	dp[n] = dp[n-1]，上述条件不满足

最终返回dp[c.length-1]



使用滚动数组优化后：

```
class Solution {
    public int translateNum(int num) {
        if(num<10) return 1;
        char[] c = String.valueOf(num).toCharArray();
        int p = 1, q = ((c[0]-'0')*10+c[1]-'0'<26)? 2:1;
        for(int i = 2; i < c.length; i++){
            int tmp = q;
            q = ((c[i-1]!='0')&&((c[i-1]-'0')*10+c[i]-'0'<26))? p+q:q;
            p = tmp;
        }
        return q;
    }
}
```



#### [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> dic = new HashMap<>();
        int res = 0, tmp = 0;
        for(int j = 0; j < s.length(); j++) {
            int i = dic.getOrDefault(s.charAt(j), -1); // 获取索引 i
            dic.put(s.charAt(j), j); // 更新哈希表
            tmp = tmp < j - i ? tmp + 1 : j - i; // dp[j - 1] -> dp[j]
            res = Math.max(res, tmp); // max(dp[j - 1], dp[j])
        }
        return res;
    }
}
```

j为当前索引，i为s[j-1]到s[0]中，s[j]这个字符最后出现的索引，dp[k]记为包含s[k]的最长不重复子字符串，因此

dp[j] = dp[j-1] + 1，当dp[j-1]不包含s[i]，即dp[j-1]<j-i;

dp[i] = j - i，当dp[j-1]包含s[i]，即dp[j-1]>=j-i；



#### [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

```
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        ListNode pre = null, cur = head;
        while(cur!=null && cur.val!=val){
            pre = cur;
            cur = cur.next;
        }
        if(pre==null){
            head = cur.next;
        }else if(cur!=null){
            pre.next = cur.next;
        }
        return head;
    }
}
```

双指针，注意处理头结点需要删除的情况即可



#### [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

```
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode low = head, fast = head;
        while(--k>=0) fast = fast.next;
        while(fast!=null) {
            fast = fast.next;
            low = low.next;
        }
        return low;
    }
}
```

双指针，快指针先移动k步，然后慢指针和快指针一起移动



#### [剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

```
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0), cur = head;
        while(l1!=null && l2!=null){
            if(l1.val<l2.val){
                cur.next = l1;
                l1 = l1.next;
            }else{
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        cur.next = (l1==null)? l2:l1;
        return head.next;
    }
}
```

用伪头部，然后分别比较



#### [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

```
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A = headA, B = headB;
        while(A!=B){
            A = (A==null)? headB:A.next;
            B = (B==null)? headA:B.next;
        }
        return A;
    }
}
```

双指针，A、B同时开始遍历，A遍历完headA后遍历headB，B遍历完headB后遍历headA，A和B第一次相等要么为相同尾部的第一个结点，要么为null（无相同尾部）



#### [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

```
class Solution {
    public int[] exchange(int[] nums) {
        int i = 0, j = nums.length - 1, tmp;
        while(i < j) {
        	//找左边第一个偶数
            while(i < j && (nums[i] & 1) == 1) i++;
            //找右边第一个奇数
            while(i < j && (nums[j] & 1) == 0) j--;
            tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }
        return nums;
    }
}
```

双指针



#### [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

方法一：set

```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Set<Integer> set = new HashSet<>();
        for(int i = 0; i < nums.length; i++){
            if(set.contains(target-nums[i])) return new int[]{nums[i], target-nums[i]};
            set.add(nums[i]);
        }
        return new int[0];
    }
}
```

方法二：由于为排序，可以用双指针

```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int i = 0, j = nums.length - 1;
        while(i < j) {
            int s = nums[i] + nums[j];
            if(s < target) i++;
            else if(s > target) j--;
            else return new int[] { nums[i], nums[j] };
        }
        return new int[0];
    }
}
```



#### [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

```
class Solution {
    public String reverseWords(String s) {
        s = s.trim(); // 删除首尾空格
        int j = s.length() - 1, i = j;
        StringBuilder res = new StringBuilder();
        while(i >= 0) {
            while(i >= 0 && s.charAt(i) != ' ') i--; // 搜索首个空格
            res.append(s.substring(i + 1, j + 1) + " "); // 添加单词
            while(i >= 0 && s.charAt(i) == ' ') i--; // 跳过单词间空格
            j = i; // j 指向下个单词的尾字符
        }
        return res.toString().trim(); // 转化为字符串并返回
    }
}
```

去除首尾空格，然后双指针，再去除首尾空格



#### [剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

```
class Solution {
    public boolean exist(char[][] board, String word) {
        char[] wordChar = word.toCharArray();
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                if(DFS(board, wordChar, i, j, 0)) return true;
            }
        }
        return false;
    }

    private boolean DFS(char[][] board, char[] wordChar, int i, int j, int index){
        if(index==wordChar.length) return true;
        if(i<0 || i>board.length-1 || j<0 || j>board[0].length-1 || wordChar[index]!=board[i][j]) return false;
        board[i][j] = '\0';
        boolean res = DFS(board, wordChar, i-1, j, index+1) || DFS(board, wordChar, i+1, j, index+1) 
                    || DFS(board, wordChar, i, j-1, index+1) || DFS(board, wordChar, i, j+1, index+1);
        board[i][j] = wordChar[index];
        return res;
    }
}
```

对于矩阵中的每一个元素用DFS到四个方向，DFS过程中注意防止重复。



#### [剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

```
class Solution {
    int m, n, k;
    boolean[][] visited;
    public int movingCount(int m, int n, int k) {
        this.m = m; this.n = n; this.k = k;
        this.visited = new boolean[m][n];
        return dfs(0, 0, 0, 0);
    }
    public int dfs(int i, int j, int si, int sj) {
        if(i >= m || j >= n || k < si + sj || visited[i][j]) return 0;
        visited[i][j] = true;
        return 1 + dfs(i + 1, j, (i + 1) % 10 != 0 ? si + 1 : si - 8, sj) + dfs(i, j + 1, si, (j + 1) % 10 != 0 ? sj + 1 : sj - 8);
    }
}
```

数位和增量公式：设 x 的数位和为 s_x ， x+1 的数位和为 s_x+1，即下一个数的数位和可以根据上一个数得出 ；

​	(x+1)%1=0时，s_x+1 = s_x - 8,

​	(x+1)%1!=0时，s_x+1 = s_x + 1,

只需要向右或向下走，就能访问所有可达解（[剑指 Offer 13. 机器人的运动范围（ 回溯算法，DFS / BFS ，清晰图解） - 机器人的运动范围 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/solution/mian-shi-ti-13-ji-qi-ren-de-yun-dong-fan-wei-dfs-b/)）；

因此可以用DFS或者BFS；



#### [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

```
class Solution {
    LinkedList<List<Integer>> res = new LinkedList<>();
    LinkedList<Integer> path = new LinkedList<>(); 
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        recur(root, sum);
        return res;
    }
    void recur(TreeNode root, int tar) {
        if(root == null) return;
        path.add(root.val);
        tar -= root.val;
        if(tar == 0 && root.left == null && root.right == null)
            res.add(new LinkedList(path));
        recur(root.left, tar);
        recur(root.right, tar);
        path.removeLast();
    }
}
```

回溯



#### [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

```
class Solution {
    private Node pre, head;
    public Node treeToDoublyList(Node root) {
        if(root == null) return null;
        dfs(root);
        pre.right = head;
        head.left = pre;
        return head;
    }
    private void dfs(Node cur){
        if(cur==null) return;
        dfs(cur.left);
        if(pre==null) {
            head = cur;
        }else{
            pre.right = cur;
        }
        cur.left = pre;
        pre = cur;
        dfs(cur.right);
    }
}
```

二叉搜索树的中序遍历遍历为有序，同时维护一个pre，用于连接相邻结点的指针



#### [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

```
class Solution {
    private int dep;

    public int kthLargest(TreeNode root, int k) {
        dep = k;
        return dfs(root);
    }
    private Integer dfs(TreeNode cur){
        if(cur==null) return null;
        Integer res = dfs(cur.right);
        if(res==null){
            if((--dep)==0) return cur.val;
            return dfs(cur.left);
        }else{
            return res;
        }
    }
}
```

右边优先的中序遍历



#### [剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

```
class Solution {
    public String minNumber(int[] nums) {
        String[] numString = new String[nums.length];
        for(int i = 0; i < nums.length; i++)
            numString[i] = String.valueOf(nums[i]);
        Arrays.sort(numString, (x,y)->(x+y).compareTo(y+x));
        StringBuilder sb = new StringBuilder();
        for(String s : numString)
            sb.append(s);
        return sb.toString();
    }
}
```

xy>yx（字符串拼接后代表的数字大小），则x"大于"y，这个排序方法具有传递性，因此将数组排序后，按序添加即可。



#### [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

```
class Solution {
    public boolean isStraight(int[] nums) {
        Arrays.sort(nums);
        int count0 = 0;
        while(nums[count0]==0) count0++;
        for(int i = count0+1; i < nums.length; i++){
            if(nums[i]-nums[i-1]>1) count0 -= (nums[i]-nums[i-1]-1);
            if(count0<0 || nums[i]==nums[i-1]) return false;
        }
        return true;
    }
}
```

0可以作为任意数字用，比如当数组为0，2，4，5，6时，0可以作为3使得数组连续

故可以先排序，找出为0的个数，然后消减0个数，判断是否连续



#### [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

```
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k >= arr.length) return arr;
        return quickSort(arr, k, 0, arr.length - 1);
    }
    private int[] quickSort(int[] arr, int k, int l, int r) {
        int i = l, j = r;
        while (i < j) {
            while (i < j && arr[j] >= arr[l]) j--;
            while (i < j && arr[i] <= arr[l]) i++;
            swap(arr, i, j);
        }
        swap(arr, i, l);
        if (i > k) return quickSort(arr, k, l, i - 1);
        if (i < k) return quickSort(arr, k, i + 1, r);
        return Arrays.copyOf(arr, k);
    }
    private void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
```

快排时，利用哨兵判断排序哪一部分

另一种方法是用大根堆，每次比较时如果堆容量小于k，那么offer，否则与堆顶元素比较大小再offer；

#### [剑指 Offer 41. 数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

```
class MedianFinder {

    Queue<Integer> A, B;

    /** initialize your data structure here. */
    public MedianFinder() {
        A = new PriorityQueue<>((x,y)->y-x);
        B = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        if(A.size()!=B.size()){
            A.add(num);
            B.add(A.poll());
        }else{
            B.add(num);
            A.add(B.poll());
        }
    }
    
    public double findMedian() {
        return A.size() != B.size() ? A.peek() : (A.peek() + B.peek()) / 2.0;
    }
}
```

两个堆A,B，分别存放数组的前一半（大根堆），后一半（小根堆），添加时进行维护；

约定A的size>=B的size



#### [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

```
class Solution {
    public int maxDepth(TreeNode root) {
        return root==null? 0 : Math.max(maxDepth(root.left)+1, maxDepth(root.right)+1);
    }
}
```

递归



#### [剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

```
class Solution {
    public boolean isBalanced(TreeNode root) {
        return recur(root) != -1;
    }

    private int recur(TreeNode root) {
        if (root == null) return 0;
        int left = recur(root.left);
        if(left == -1) return -1;
        int right = recur(root.right);
        if(right == -1) return -1;
        return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
    }
}
```

后序遍历+剪枝



#### [剑指 Offer 64. 求1+2+…+n](https://leetcode-cn.com/problems/qiu-12n-lcof/)

```
class Solution {
    public int sumNums(int n) {
        boolean x = n > 1 && (n += sumNums(n - 1)) > 0;
        return n;
    }
}
```

使用&&终止递归



#### [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

```
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null) return null;
        if(root.val>p.val && root.val>q.val) return lowestCommonAncestor(root.left, p, q);
        if(root.val<p.val && root.val<q.val) return lowestCommonAncestor(root.right, p, q);
        return root;
    }
}
```

利用二叉搜索树的性质



#### [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

```
class Solution {
    /**
     * 二叉树的最近公共祖先
     * 思路：
     * 三种情况：
     * 1、p q 一个在左子树 一个在右子树 那么当前节点即是最近公共祖先
     * 2、p q 都在左子树 
     * 3、p q 都在右子树
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            // p q 一个在左，一个在右
            return root;
        }
        if (left != null) {
            // p q 都在左子树
            return left;
        }
        if (right != null) {
            // p q 都在右子树
            return right;
        }
        return null;
    }
}
```



#### [剑指 Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

```
class Solution {
    private int[] preorder;
    private Map<Integer,Integer> dic = new HashMap<>();

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        for(int i=0; i<inorder.length; i++) dic.put(inorder[i], i);
        return recur(0, 0, inorder.length-1);
    }

    private TreeNode recur(int root, int left, int right){
        if(left>right) return null;
        TreeNode cur = new TreeNode(preorder[root]);
        int index = dic.get(preorder[root]);
        cur.left = recur(root+1, left, index-1);
        cur.right = recur(root + index - left + 1, index + 1, right);
        return cur;
    }
}
```

分治法：建立当前结点后，建立左右子树

recur函数意义

​	root:根节点在先序遍历的位置

​	left子树在中序遍历的左边界，right子树在中序遍历的右边界



#### [剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

快速幂（矩阵的）？



#### [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

```
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return recur(postorder, 0, postorder.length-1);
    }

    private boolean recur(int[] postorder, int i, int j){
        if(i>=j) return true;
        int k = i;
        while(postorder[k]<postorder[j]) k++;
        int p = k;
        while(postorder[p]>postorder[j]) p++;
        return p==j && recur(postorder, i, k-1) && recur(postorder, k, j-1);
    }
}
```

分治，i为当前数在后序遍历的左边界，j为当前树在后序遍历的右边界

从i开始遍历，找到左右树边界，然后遍历右树判断是否满足都小于当前根，然后分治



#### [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

```
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int res = 0;
        while(n!=0){
            ++res;
            n &= (n-1);
        }
        return res;
    }
}
```



#### [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

```
class Solution {
    public int add(int a, int b) {
        while(b != 0) { // 当进位为 0 时跳出
            int c = (a & b) << 1;  // c = 进位
            a ^= b; // a = 非进位和
            b = c; // b = 进位
        }
        return a;
    }
}
```

异或相当于无进位求和

&相当于求每位数的进位



#### [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

```
class Solution {
    public int[] singleNumbers(int[] nums) {
        int x = 0, y = 0, n = 0, m = 1;
        for(int num : nums)               // 1. 遍历异或
            n ^= num;
        while((n & m) == 0)               // 2. 循环左移，计算 m
            m <<= 1;
        for(int num: nums) {              // 3. 利用m对nums分组，将两个不同的数分开
            if((num & m) != 0) x ^= num;  // 4. 当 num & m != 0
            else y ^= num;                // 4. 当 num & m == 0
        }
        return new int[] {x, y};          // 5. 返回出现一次的数字
    }
}
```

利用异或的性质，两个相同的数异或为0



#### [剑指 Offer 56 - II. 数组中数字出现的次数 II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

```
class Solution {
    public int singleNumber(int[] nums) {
        int[] k = new int[32];
        for (int num : nums) {
            for (int i = 0; i < 32; i++) {
                k[i] += num & 1;
                num >>= 1;
            }
        }
        int res = 0;
        for (int i = 0; i < 32; i++) {
             res |= (k[i] % 3) << i;
        }
        return res;
    }
}
```

出现3次的数的二进制的每1位的和，能够被3整除。如果某1位能够被3整除，那么在要找的数对应位为0，否则为1.



#### [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

```
class Solution {
    public int majorityElement(int[] nums) {
        int x = 0, votes = 0;
        for(int num : nums){
            if(votes == 0) x = num;
            votes += num == x ? 1 : -1;
        }
        return x;
    }
}
```

投票法

[剑指 Offer 39. 数组中出现次数超过一半的数字（摩尔投票法，清晰图解） - 数组中出现次数超过一半的数字 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/solution/mian-shi-ti-39-shu-zu-zhong-chu-xian-ci-shu-chao-3/)



#### [剑指 Offer 66. 构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

```
class Solution {
    public int[] constructArr(int[] a) {
        if(a.length==0) return new int[0];
        int[] res = new int[a.length];
        res[0] = 1;
        int tmp = 1;
        for(int i=1; i<a.length; i++){
            res[i] = res[i-1]*a[i-1];
        }
        tmp = 1;
        for(int i=a.length-1; i>0; i--){
            tmp *= a[i];
            res[i-1] *= tmp;
        }
        return res;
    }
}
```

每个B[i]对应一个横轴，每个A[i]对应一个纵轴，依次求下三角，上三角即可



#### [剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

```
class Solution {
    public int cuttingRope(int n) {
        /*
        dp五部曲:
        1.状态定义:dp[i]为长度为i的绳子剪成m段最大乘积为dp[i]
        2.状态转移:dp[i]有两种途径可以转移得到
            2.1 由前一个dp[j]*(i-j)得到,即前面剪了>=2段,后面再剪一段,此时的乘积个数>=3个
            2.2 前面单独成一段,后面剩下的单独成一段,乘积为i*(i-j),乘积个数为2
            两种情况中取大的值作为dp[i]的值,同时应该遍历所有j,j∈[1,i-1],取最大值
        3.初始化:初始化dp[1]=1即可
        4.遍历顺序:显然为正序遍历
        5.返回坐标:返回dp[n]
        */
        // 定义dp数组
        int[] dp = new int[n + 1];
        // 初始化
        dp[1] = 1;  // 指长度为1的单独乘积为1
        // 遍历[2,n]的每个状态
        for(int i = 2; i <= n; i++) {
            for(int j = 1; j <= i - 1; j++) {
                // 求出两种转移情况(乘积个数为2和2以上)的最大值
                int tmp = Math.max(dp[j] * (i - j), j * (i - j));
                dp[i] = Math.max(tmp, dp[i]);
            }
        }
        return dp[n];
    }
}
```



#### [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

```
class Solution {
    public int[][] findContinuousSequence(int target) {
        int i = 1, j = 2, s = 3;
        List<int[]> list = new ArrayList<>();
        while(i<j){
            if(s==target){
                int[] tmp = new int[j-i+1];
                for(int k=i; k<=j; k++) tmp[k-i] = k;
                list.add(tmp);
            }
            if(s>=target) s -= i++;
            else s += ++j;
        }
        return list.toArray(new int[0][]);
    }
}
```

滑动窗口



#### [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

```
class Solution {
    public int lastRemaining(int n, int m) {
        int x = 0;
        for (int i = 2; i <= n; i++) {
            x = (x + m) % i;
        }
        return x;
    }
}
```

约瑟夫环：

已知（n-1,m）的解dp(n-1)，可以求得（n,m）的解dp(n)

(n,m)删除一个（m-1）%n后，下个数字为（m%n）,所有数字与n-1问题数字的对应关系：

​	n-1,m问题     -》   n,m问题 删除后

​	0					 -》	（m%n）+0	

...

因此f(n) = (f(n-1)+m%n)%n = (f(n-1)+m)%n



#### [剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

```
class Solution {
    public int[] spiralOrder(int[][] matrix) {
        if(matrix.length == 0) return new int[0];
        int l = 0, r = matrix[0].length - 1, t = 0, b = matrix.length - 1, x = 0;
        int[] res = new int[(r + 1) * (b + 1)];
        while(true) {
            for(int i = l; i <= r; i++) res[x++] = matrix[t][i]; // left to right.
            if(++t > b) break;
            for(int i = t; i <= b; i++) res[x++] = matrix[i][r]; // top to bottom.
            if(l > --r) break;
            for(int i = r; i >= l; i--) res[x++] = matrix[b][i]; // right to left.
            if(t > --b) break;
            for(int i = b; i >= t; i--) res[x++] = matrix[i][l]; // bottom to top.
            if(++l > r) break;
        }
        return res;
    }
}
```

模拟



#### [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

```
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        int i = 0;
        for(int num : pushed) {
            stack.push(num); // num 入栈
            while(!stack.isEmpty() && stack.peek() == popped[i]) { // 循环判断与出栈
                stack.pop();
                i++;
            }
        }
        return stack.isEmpty();
    }
}
```

使用辅助栈进行模拟，如果入栈，栈顶元素和出栈序列相等，那么弹出



#### [剑指 Offer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

不看

#### [剑指 Offer 67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

不看



#### [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

```
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0 || k == 0) return new int[0];
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[nums.length - k + 1];
        for(int j = 0, i = 1 - k; j < nums.length; i++, j++) {
            // 删除 deque 中对应的 nums[i-1]
            if(i > 0 && deque.peekFirst() == nums[i - 1])
                deque.removeFirst();
            // 保持 deque 递减
            while(!deque.isEmpty() && deque.peekLast() < nums[j])
                deque.removeLast();
            deque.addLast(nums[j]);
            // 记录窗口最大值
            if(i >= 0)
                res[i] = deque.peekFirst();
        }
        return res;
    }
}
```

维护一个非严格递减单调队列（双端），队列里边的元素都是当前窗口中的元素



#### [剑指 Offer 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

同上，需要维护一个非严格递减的单调队列（双端），还有一个正常队列



#### [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

```
public class Codec {

    public String serialize(TreeNode root) {
        if(root==null) return "[]";
        StringBuilder sb = new StringBuilder("[");
        Queue<TreeNode> q = new LinkedList<TreeNode>(){{add(root);}};
        while(!q.isEmpty()){
            TreeNode n = q.poll();
            if(n!=null){
                sb.append(n.val+",");
                q.add(n.left);
                q.add(n.right);
            }else{
                sb.append("null,");
            }
        }
        sb.deleteCharAt(sb.length()-1);
        sb.append("]");
        return sb.toString();
    }

    public TreeNode deserialize(String data) {
        if(data=="[]") return null;
        String[] s = data.substring(1, data.length() - 1).split(",");
        TreeNode root = new TreeNode(Integer.parseInt(s[0]));
        Queue<TreeNode> q = new LinkedList<TreeNode>() {{ add(root); }};
        int i = 1;
        while(!q.isEmpty()){
            TreeNode n = q.poll();
            if(!s[i].equals("null")){
                n.left = new TreeNode(Integer.parseInt(s[i]));
                q.add(n.left);
            }
            i++;
            if(!s[i].equals("null")){
                n.right = new TreeNode(Integer.parseInt(s[i]));
                q.add(n.right);
            }
            i++;
        }
        return root;
    }
}
```

序列化的时候借助队列使用BFS层序遍历；反序列化的时候借助队列BFS（特别要注意i）



#### [剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

```
class Solution {
    List<String> res = new LinkedList<>();
    char[] c;
    public String[] permutation(String s) {
        c = s.toCharArray();
        dfs(0);
        return res.toArray(new String[res.size()]);
    }
    void dfs(int x) {
        if(x == c.length - 1) {
            res.add(String.valueOf(c));      // 添加排列方案
            return;
        }
        HashSet<Character> set = new HashSet<>();
        for(int i = x; i < c.length; i++) {
            if(set.contains(c[i])) continue; // 重复，因此剪枝
            set.add(c[i]);
            swap(i, x);                      // 交换，将 c[i] 固定在第 x 位
            dfs(x + 1);                      // 开启固定第 x + 1 位字符
            swap(i, x);                      // 恢复交换
        }
    }
    void swap(int a, int b) {
        char tmp = c[a];
        c[a] = c[b];
        c[b] = tmp;
    }
}
```

dfs排列树，用set防止排列过程中选择了重复的元素



#### [剑指 Offer 19. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)



#### [剑指 Offer 49. 丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

```reasonml
class Solution {
    public int nthUglyNumber(int n) {
        int[] dp = new int[n];  // 使用dp数组来存储丑数序列
        dp[0] = 1;  // dp[0]已知为1
        int a = 0, b = 0, c = 0;    // 下个应该通过乘2来获得新丑数的数据是第a个， 同理b, c

        for(int i = 1; i < n; i++){
            // 第a丑数个数需要通过乘2来得到下个丑数，第b丑数个数需要通过乘2来得到下个丑数，同理第c个数
            int n2 = dp[a] * 2, n3 = dp[b] * 3, n5 = dp[c] * 5;
            dp[i] = Math.min(Math.min(n2, n3), n5);
            if(dp[i] == n2){
                a++; // 第a个数已经通过乘2得到了一个新的丑数，那下个需要通过乘2得到一个新的丑数的数应该是第(a+1)个数
            }
            if(dp[i] == n3){
                b++; // 第 b个数已经通过乘3得到了一个新的丑数，那下个需要通过乘3得到一个新的丑数的数应该是第(b+1)个数
            }
            if(dp[i] == n5){
                c++; // 第 c个数已经通过乘5得到了一个新的丑数，那下个需要通过乘5得到一个新的丑数的数应该是第(c+1)个数
            }
        }
        return dp[n-1];
    }
}
```

三指针；

a表示前(a-1)个数都已经乘过一次2了，下次应该乘2的是第a个数；b表示前(b-1)个数都已经乘过一次3了，下次应该乘3的是第b个数；c表示前(c-1)个数都已经乘过一次5了，下次应该乘5的是第c个数；



#### [剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

```
public double[] dicesProbability(int n) {
        double res[] = new double[n*5+1];
        double dp[][] = new double[n+1][n*6+1];
        for(int i = 1;i <= 6;i++){
            dp[1][i] = 1.0/6;
        }
        for(int i = 2;i <= n;i++){
            for(int j = i;j <= i*6;j++){
            	//第i-1的基础上，第i个骰子点数可以为1-6
                for(int k = 1;k <= 6;k++){
                    if(j-k > 0)
                        dp[i][j] += dp[i-1][j-k]/6;
                    else
                        break;
                }
            }
        }
        for(int i = 0;i <= 5*n;i++){
            res[i] = dp[n][n+i];
        }
        return res;
    }
```

dp[i] [j]表示i个骰子，和为j的概率；

dp[i]可以由dp[i-1]转移而来，假设第i个骰子点数为k(1-6)，则dp[i] [j] = dp[i-1] [j-k]/6 (k为1-6)



#### [剑指 Offer 17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)（大数打印）

```
class Solution {
    int[] res;
    int nine = 0, count = 0, start, n;
    char[] num, loop = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    public int[] printNumbers(int n) {
        this.n = n;
        res = new int[(int)Math.pow(10, n) - 1];
        num = new char[n];
        start = n - 1;
        dfs(0);
        return res;
    }
    void dfs(int x) {
        if(x == n) {
            String s = String.valueOf(num).substring(start);
            if(!s.equals("0")) res[count++] = Integer.parseInt(s);
            if(n - start == nine) start--;
            return;
        }
        for(char i : loop) {
            if(i == '9') nine++;
            num[x] = i;
            dfs(x + 1);
        }
        nine--;
    }
}
```

short，int，long等表示的数的范围是有限的，因此考虑大数时，需要用字符串来表示数；

数是0-9的全排列，可以用分治法处理每一位，以（0-9）生成；

生成的字符串会出现多余的前导0，因此需要用添加进结果集时，需要截去前导0，故维护一个start作为左边界，维护方法如下：

​	当发生进位比如99到100时，左边界需要-1，nine作为字符串中9的个数，如果n - start = nine，表示此	时需要进位，则start--；



#### [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

```
class Solution {
    int[] nums, tmp;
    public int reversePairs(int[] nums) {
        this.nums = nums;
        tmp = new int[nums.length];
        return mergeSort(0, nums.length - 1);
    }
    private int mergeSort(int l, int r) {
        // 终止条件
        if (l >= r) return 0;
        // 递归划分
        int m = (l + r) / 2;
        int res = mergeSort(l, m) + mergeSort(m + 1, r);
        // 合并阶段
        int i = l, j = m + 1;
        for (int k = l; k <= r; k++)
            tmp[k] = nums[k];
        for (int k = l; k <= r; k++) {
            if (i == m + 1)
                nums[k] = tmp[j++];
            else if (j == r + 1 || tmp[i] <= tmp[j])
                nums[k] = tmp[i++];
            else {
                nums[k] = tmp[j++];
                res += m - i + 1; // 统计逆序对
            }
        }
        return res;
    }
}
```

归并排序中，在归并的时候计算逆序对；



#### [剑指 Offer 14- II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

```
class Solution {
    public int cuttingRope(int n) {
        if(n <= 3) return n - 1;
        long res=1L;
        int p=(int)1e9+7;
        //贪心算法，优先切三，其次切二
        while(n>4){
            res=res*3%p;
            n-=3;
        }
        //出来循环只有三种情况，分别是n=2、3、4
        return (int)(res*n%p);
    }
}
```

优先切3，最后剩下一段



#### [剑指 Offer II 003. 前 n 个数字二进制中 1 的个数](https://leetcode-cn.com/problems/w3tCBm/)

方法一：

```
class Solution {
    public int[] countBits(int n) {
        int[] bits = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            bits[i] = countOnes(i);
        }
        return bits;
    }

    public int countOnes(int x) {
        int ones = 0;
        while (x > 0) {
            x &= (x - 1);
            ones++;
        }
        return ones;
    }
}
```

遍历数组，对于每一个数单独计算；计算方法为：对于任意整数 x，令 x=x & (x-1)，该运算将 x 的二进制表示的最后一个 1 变成 0。因此，对 x 重复该操作，直到 x 变成 0，则操作次数即为 x 的「一比特数」。

方法二：

动态规划+位运算

[让你秒懂的双百题解！ - 前 n 个数字二进制中 1 的个数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/w3tCBm/solution/rang-ni-miao-dong-de-shuang-bai-ti-jie-b-84hh/)



#### [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

```
class Solution {
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        long count = 9;
        //循环出来后digit为n所处的数位；start为此数位开始的数字；n为n所处的数字偏移start
        while (n > count) { // 1.
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        // 根据偏移得到n对应的数字
        long num = start + (n - 1) / digit; // 2.
        // 得到n对应的位
        return Long.toString(num).charAt((n - 1) % digit) - '0'; // 3.
    }
}
```

