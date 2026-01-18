**语法部分**
**一。输入**：
1.一次性读入
```python
date=sys.stdin.read().split()  
interator=iter(date)
try:
	for i in range(n):
		m=int(next(interator))
		......
	or
	while True:
		......
except StopIteration
	pass
```
2.
将[1,2,3]变为123：int("".join(str(x) for x in lst))
将输入123变为[1,2,3]:nums=[int(i) for i in input()]
**二。格式化**
```python

# 输出
格式化：f-string方法
1.x = 3.14159
print(f"x 的平方是 {x ** 2:.2f}")   # 保留两位小数
"x 的平方是 9.87"
2.支持表达式
a, b = 5, 3
print(f"{a} + {b} = {a + b}")
print(f"{a} * {b} = {a * b}")
print(f"平均值是 {(a + b) / 2}")
3.小数点控制
pi = 3.1415926
print(f"π = {pi:.2f}")  # 保留两位小数
"π = 3.14"

```
**三。函数**
```python
1. 数学计算：
    - `pow(x, y)`：计算x的y次方
    - `math.floor()`：向下取整
    - `math.ceil()`：向上取整
2. 列表操作：
    - `extend()`：在列表末尾一次性追加另一个序列中的多个值
      将nums从start开始的数字加到result里面：result.extend(nums[start:])
    - `insert()`：将对象插入列表
    - i,arr=enumerate()
    -删除：
    列表按索引删除 del mylist[2]不返回新列表，mylist.pop(i)返回删除的元素，切片返回新列表
3. 字符串操作：
    - `str.replace()`：字符串替换
    - `str.split()`：字符串分割
    - `str.strip()`：移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    - `str.join()`：用于将序列中的元素以指定的字符连接生成一个新的字符串
    - `enumerate()`：为迭代器增加索引标签，常用于获取列表的元素及其索引
```
**四。排序：**
按照第一个元素排序：lst.sort(key=lambda x: x[0])
降序排序：lst.sort(key=lambda x: x[1], reverse=True)
s=sorted(thelist)返回新列表

**五。缓存**
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

**六。string 与ASCII码的用法**
**七。集成语句：**
1.maxlen=len(max(thelist,key=lambda x : len(x)))),求出一个列表中最长元素的长度
	 先定义函数lambdax表示对于x取长度，然后对于可迭代对象thelist中的元素计算lambda x，并求得最大的lambda x对应的x,最后再对于x取len
2.list.sort(key=lambda x: x* *2 ),用法同1

**算法部分**
**二分查找模版：**
```python
while left<=right:  
    medium=(left+right)//2  
    flag=query(medium)  
    if flag==0:  
        break  
    elif flag==1:  
        right=medium-1  
    else:  
        left=medium+1
```
规避所有边界问题，包括left,right本身是目标的情形


**最大最小整数**：将181 18 987 234拼接为最大与最小整数：
关键问题：出现前几位重复的处理办法
法1：冒泡排序：对于每一个i去枚举j，判断i+j与j+i谁大并换序
```python
# O(n^2)
n = int(input())
nums = input().split()
for i in range(n - 1):
    for j in range(i+1, n):
        #print(i,j)
        if nums[i] + nums[j] < nums[j] + nums[i]:
            nums[i], nums[j] = nums[j], nums[i]

ans = "".join(nums)
nums.reverse()
print(ans + " " + "".join(nums))
```
法二：扩展字符串长度：如果出现前几位重复那么就通过自我重复形成长度相同的字符串进行比较
```python
# 两倍长度是正确的。O(nlogn)
from math import ceil
input()
lt = input().split()

max_len = len(max(lt, key = lambda x:len(x)))
lt.sort(key = lambda x: x * ceil(2*max_len/len(x)))# 重复子串
lt1 = lt[::-1]
print(''.join(lt1),''.join(lt))
```
精妙之处：如果在已有部分没有分出大小，那么接下来相当于比较长的序列的下一段和已经比较过的上一个周期的大小，由于上一个周期刚刚比较过并且为重复周期，所以相当于与第一个重复周期进行比较。所以只要将短的单元不断重复即可。由于拼接成的字符串长度不可能超过2*lenmax，所以只需将所有字符串重复直到2*lenmax即可。


**2048游戏**：给出一系列的2的幂次，你可以合并相同的数字然后产生新数字并删除原来的两个数字。问对于给定的数据能否加出2048？
算法：
利用二进制的特点：$2^k+2^k=2^{k+1},2048=2^{11}$,
首先，如果能通过加法加出来2个1024或者自带1到2个1024，否则是不可能产生2048的
$1+2+4+⋯+1024=2047$
在这个基础上一旦在某个幂次上加入一个重复元素，就会点爆这个链直接打到2048.一旦中间出现截断，截断下面的所有数加在一起也越不过这个截断，那么截断下面的数就废了。
换言之，通过这种加法得到的取值是二进制稠密的，所以**只需判断所有比2048小的数的总和是否超过2048即可**
```python
def count(s):  
    thesum=0  
    for j in s:  
        if j<=2048:  
           thesum+=j  
    if thesum>=2048:  
        print("YES")  
    else:  
        print("NO")  
n=int(input())  
for i in range(n):  
    m=int(input())  
    s=list(map(int,input().split()))  
    count(s)
```


**垃圾炸弹**：
观察到数据点的特点：炸弹的半径可能很大，但是垃圾点不超过20个。
所以不超时的话需要枚举垃圾点在不在区间范围内而不是检查区间内的点是不是垃圾点。

高级数据结构：栈，队列，堆
注意：中间过程不要修改队列和堆的值，也不要对栈排序
操作：
1。**栈**：语法上和列表没有区别，是一种只在末端操作的列表。
	入栈：stack.append(x)
	出栈：stack.pop() 返回元素并删除
	读取最后一个数：stack[-1]
	eg:校门外的树：
    维护stack,如果stack不为空并且新的区间的左边界小于stack的最大值，那就维护右端最大值；else:将新的区间压入stack
**单调栈：维护一个有单调性的栈。**
eg1**每日温度**：维护一个单调递减的栈，保存没有遇到更大的数字的数。每次遇到一个新的数字就进行while循环检查前面栈里的数字。如果有比新的数字小的，那就把他们弹出去并记录答案。如果没有就跳过。最后再将新的数字压入栈，作为最新的没有遇到更大的数字的数。
```python
temperatures=[73,74,75,71,69,72,76,73]  
n=len(temperatures)  
stack=[]  
answer=[0]*n  
for i in range(n):  
    while stack and temperatures[i]>temperatures[stack[-1]]:  
        top=stack.pop()  
        answer[top]=i-top  
    stack.append(i)  
return answer
```
eg2:**接雨水**：给定n作为数据个数，然后给出一个列表height来表示高度。求最多可以接多少水
法一：双指针：从左右两端进行，按一列一列填水，每个位置一次到位
```python
n=int(input())  
height=list(map(int,input().split()))  
left=0  
right=n-1  
left_max=0  
right_max=0  
result=0  
while left<right:  
    if height[left]<height[right]:  
        if height[left]>=left_max:  
            left_max=height[left]  
        else:  
            result+=left_max-height[left]  
        left+=1  
    else:  
        if height[right]>=right_max:  
            right_max=height[right]  
        else:  
            result+=right_max-height[right]  
        right-=1  
print(result)
```
维护一个left左侧极大值和right右侧极大值。
策略如下：如果height(left)<height(right),且height(left)<left_max。那就说明left一定可以接水，并且高度为left_max-height(left).(这是因为left向右移动的条件是height(right)>height(left),所以右侧一定存在一个比左侧left_max还高的点)；如果height(left)>=left_max,那么更新最大值并且不接水。这两种情况均要求left右移一格。right指针的操作与left完全对称。While的条件不取等以保证不会死循环。
法二：单调栈：通过维护单调递减栈来一行一行填水，每个位置不能被一次性填完

```python
n=6  
height=[0,1,0,2,1,0,1,3,2,1,2,1]  
#单调栈算法。维护一个单调递减的栈。每次从右方弹出比目标小的元素然后压入新的元素。  
#然后每次弹出元素的时候计算覆盖水的宽度并计算水的量  
stack=[]  
water=0  
for i in range(len(height)):  
    while stack and height[i]>height[stack[-1]]: #找到一个上升的墙壁
        last=stack.pop()  #弹出底部
        down=height[last]  
        if not stack:  #如果左边没有元素了那就说明无法形成v形结构
            break  
        left=stack[-1]  #左侧墙壁
        h=min(height[left],height[i])-down  #水的深度
        w=i-left-1  #水的宽度
        water+=h*w  
    stack.append(i)  #无论是否作为右墙壁均入栈
print(water)
```
eg3:**辅助栈**：通过多建立一个单调栈来便捷的给出最值
堆猪游戏：、
		1. push n n是整数(0<=0 <=20000)，表示叠上一头重量是n斤的新猪
		2. pop 表示将猪堆顶的猪赶走。如果猪堆没猪，就啥也不干
		3. min 表示问现在猪堆里最轻的猪多重。如果猪堆没猪，就啥也不干
```python
pig, pigmin = [], []
while True:
    try:
        *line, = input().split()
        if "pop" in line:
            if len(pig) == 0:
                continue

            val = pig.pop()
            if len(pigmin) > 0 and val == pigmin[-1]:
                pigmin.pop()
        elif "push" in line:
            val = int(line[1])
            pig.append(val)
            if len(pigmin) == 0 or val <= pigmin[-1]:
                pigmin.append(val)
        elif "min" in line:
            if len(pig) == 0:
                continue
            else:
                print(pigmin[-1])
    except EOFError:
        break
```
引入辅助栈pigmin,只在新加入的元素比目前pigmin[-1]还要小的时候或者pigmin已经没有元素的时候将它压入辅助栈。当pig执行pop操作时，只有pop出的元素正好是pigmin[-1]时对于pigmin进行pop.而pigmin能够不漏掉所有极小值的原因在于压入顺序：后压入pig的数一定先被pop，所以无需入栈除非它是所有值里面最小的。
2.**队列**：只比列表多了一个左侧弹出的操作，允许两端操作
	引入：from collections import deque
	        queue=deque([1,2,3,4,5])
	入队：queue.append(x)
	出队：队尾：queue.pop()
		   队首：queue.popleft()
	读取元素：queue[-1]/queue[0]

eg2**滑动窗口最大值**：实时返回序列中滑动窗口内的最大值
```python
nums=[1,3,-1,-3,5,3,6,7]  
k=3  
from collections import deque  
result=[]  
#维护最大值候选序列  
queue=deque([])#保存递增的索引，并且索引对应的nums递减  
n=len(nums)  
for i in range(n):  
    while queue and nums[queue[-1]]<nums[i]:  
        queue.pop()  
    queue.append(i)  
    while i-queue[0]+1>k:  
        queue.popleft()  
    if i>=k-1 and i-queue[0]+1<=k:  
        result.append(nums[queue[0]])  
print(result)
```
维护一个单调队列，其储存候选最大值的索引。注意到窗口最右面的数的左侧比他还小的数是没有机会成为最大值的。所以直接利用第一个while将他们pop走，然后将新的数字压入队列。为了维护窗口长度，利用第二个while从左侧将超出范围的索引popleft。一旦形成合法窗口（两个条件：一个是i需要满足形成第一个窗口，另一个是候选名单长度比窗口小），就向result里面加入一个结果。时间复杂度仅为O(n)
3.**堆**：把一堆数扔进去，它能实时返回最小值
	引入： import heapq
	初始化：heap=[3,2,1]
			heapq.heapify(heap)
			此时heap=[1,2,3]
			**绝对不能赋值：heap=heap.heapify([3,2,1]),这样heap=None**
	入堆：heapq.heappush(heap,-1),heap=[-1,1,2,3]
	弹出堆顶最小值：heapq.heapop(heap)   弹出-1
**剪绳子**：哈夫曼树算法：将剪绳子的过程变为逆过程。计算剪绳子的花费等效于计算将短的绳子合并成长绳子的花费。可以使用贪心算法优化这个过程，每次都用堆给出最小的两个数字，然后加在一起之后再压回堆里面。这样能保证最后一定能合成长绳子并且花费最小
```python
n=int(input())  
nums=list(map(int,input().split()))  
import heapq  
heapq.heapify(nums)  
s=0  
while len(nums)>1:  
    x1=heapq.heappop(nums)  
    x2=heapq.heappop(nums)  
    s+=x1+x2  
    heapq.heappush(nums,x1+x2)  
print(s)
```
**走山路：Dijkstra+懒删除**
建立一个堆，存储（cost,i,j）,用于表示走到某个点所需要的目前最小cost
1.初始化堆：只加入（0,start[0],start[1]）
2.建立一个列表thelist用于记录算出的最小消耗，初始即为无穷大
3.从起点开始找临近的点的最小值，这条路径一定是最短的。然后从这个点开始向四周寻找新的点。将这些信息压入堆，并且弹出这次找到的点。
	**懒删除**：此处对于某个点去重消耗过多时间，因为堆不擅长查找任意元素，但是这样浪费的空间可以接受，所以进行懒删除更加划算
4.重复3的操作，每次弹出目标点之后先与thelist进行一个比较。如果之前已经比现在小，那就直接跳过不进行四周的扩展。
	  **某个点第一次被堆吐出来一定是该点的最小消耗**：这是因为如果更短的路存在，那它一定会被先弹出来进行扩展，根本等不到更长的路被弹出来。
5.一旦栈弹出end，立即终止循环（和bfs类似）
```python
import heapq  
def solve(ground,start,end):  
    thelist=[]  
    direction=[(-1,0),(1,0),(0,-1),(0,1)]  
    #初始化  
    thelist.append((0,start[0],start[1]))  
    heapq.heapify(thelist)  
    #寻找最小路径  
    mincost=[[float("inf")]*n for _ in range(m)]  
    mincost[start[0]][start[1]]=0  
    while thelist:  
        cost,x,y=heapq.heappop(thelist)  
        if x==end[0] and y==end[1]:  
            mincost[x][y]=cost  
            break  
        if mincost[x][y]>=cost:  
            mincost[x][y]=cost  
            for step in direction:  
                nextpoint=[x+step[0],y+step[1]]  
                if 0<=nextpoint[0]<=m-1 and 0<=nextpoint[1]<=n-1:  
                    newcost=abs(ground[nextpoint[0]][nextpoint[1]]-ground[x][y])+cost  
                    if newcost<mincost[nextpoint[0]][nextpoint[1]]:  
                        mincost[nextpoint[0]][nextpoint[1]]=newcost  
                        heapq.heappush(thelist,(newcost,nextpoint[0],nextpoint[1]))  
    if mincost[end[0]][end[1]]==float("inf"):  
        print("NO")  
    else:  
        print(mincost[end[0]][end[1]])  
m,n,p=map(int,input().split())#m为行数，n为列数，p为测试数  
ground=[]  
for i in range(m):  
    list_=[]  
    input_=input().split()  
    for x in input_:  
        if x=="#":  
            list_.append(float("inf"))  
        else:  
            list_.append(int(x))  
    ground.append(list_)  
for i in range(p):  
    list_=list(map(int,input().split()))  
    start=[list_[0],list_[1]]  
    end=[list_[2],list_[3]]  
    if ground[start[0]][start[1]]!=float("inf") and ground[end[0]][end[1]]!=float("inf"):  
        solve(ground,start,end)  
    else:  
        print("NO")
```
优化：1.如果可以不记录全部地图就记录联通关系，省内存 2.懒删除剪枝使用集合更快3.终点被弹出之后立即终止**注意：将元素压进堆千万不要append**
**动态规划**：
	**递推式**
		1。**flowers**：
		递推关键在于最后一次不是红花就是k个白花，然后只要i>k,就可以进行比较dp[i-k]与dp[i-1],得到递推式
		2.**最大上升子序列**：
```python
n=int(input())  
thelist=list(map(int,input().split()))  
dp=[0]*n#表示以thelist[i]为结尾的上升序列的最大长度  
dp[0]=1  
for i in range(1,n):  
    it = [1]  
    for j in range(i):  
        if thelist[j]<thelist[i]:  
            it.append(dp[j]+1)  
    dp[i]=max(it)  
print(max(dp))
```
dp储存以某个数结尾的最长上升子序列的长度，而不是某个数之前的最大上升子序列长度。不然会导致递推困难。对于每个末尾数字，枚举前面的数字的dp列表，然后比较大小更新即可。最后输出dp列表中最大的数字。
		**3.:kadane算法**：解决连续子链的最大和问题
		如果前面的和是小于零的，那就另起炉灶重新开始
		递推：currentsum=max(currentsum+nums[i],nums[i])
		       maxsum=max(maxsum,currentsum)

**多重dp：**
	eg1:股票买卖：可以在给定时间进行两次买卖，并且可以同一天买卖。找出最后的最大收益
```python
def solve(n,thelist):  
    buy1=-10000  
    buy2=-10000  
    sell1=0  
    sell2=0  
    for i in range(n):  
        buy1=max(buy1,-thelist[i])  
        buy2=max(buy2,sell1-thelist[i])  
        sell1=max(sell1,thelist[i]+buy1)  
        sell2=max(sell2,buy2+thelist[i])  
    print(sell2)  
t=int(input())  
for _ in range(t):  
    n=int(input())  
    thelist=list(map(int,input().split()))  
    solve(n,thelist)
```
建立buy1,buy2,sell1,sell2四个互相耦合的递推关系，来进行dp
	eg2:土豪购物：土豪可以连续买很多东西，但是土豪可以放回去一个东西。问土豪能够买到的最多有多少钱？
```python
thelist=list(map(int,input().strip().split(",")))  
n=len(thelist)  
if n>=1:  
    dp1=[0]*n#以某个数字结尾的最大和序列  
    currentsum=-float("inf")  
    maxsum=-float("inf")  
    for i in range(n):  
        currentsum=max(currentsum+thelist[i],thelist[i])  
        maxsum=max(maxsum,currentsum)  
        dp1[i]=currentsum  
    dp2=[0]*n#以i开头的最大和序列  
    currentsum=-float("inf")  
    for j in range(n-1,-1,-1):  
        currentsum = max(currentsum + thelist[j], thelist[j])  
        dp2[j] = currentsum  
    #如果进行了删除，枚举删除掉的元素的索引k  
    for k in range(1,n-1):  
        price=dp1[k-1]+dp2[k+1]  
        maxsum=max(maxsum,price)  
    print(maxsum)  
else:  
    print(thelist[0])
```
使用两个kadane算法，分别建立dp1与dp2,用来表示以某个数结尾的最大和以及以某个数开始的最大和。然后枚举删除的元素的索引，将两个列表加在一起即可。

**多维dp:**
1.01背包问题：小偷背包容量为B，物品数量为N，每个物品至多被偷一次。问能够偷走的最大价值。
相当于建立二维dp.第一层dp[i,j]表示在载重为i的时候，考虑了前j个物体时的最佳状态。不断考虑新的物体，也就是j->j+1,然后进行递推：如果目前的承重大于新物体的质量就进行查表更新dp[i,j]=max(dp[i-w,j-1]+cost,dp[i,j-1]),否则就继承原来的状态.
得益于问题中第二个维度是有限依赖的，具体的来说就是dp[,j]仅仅依赖于dp[,j-1],这可以让我们建立滚动数组来讲二维列表压缩为一维dp列表。完全的把对于j的更新藏在每次对于i的更新里面。
```python
N, B = map(int, input().split())
cost = list(map(int, input().split()))
weigh = list(map(int, input().split()))
dp = [0] * (B + 1)
for i in range(N):
    c = cost[i]
    w = weigh[i]
    dp1 = [0] * (B + 1)
    for j in range(1, B + 1):
        if j >= w:
            dp1[j] = max(dp[j], dp[j - w] + c)
    dp = dp1[:]
print(dp[-1])
```
2.整数划分问题：将一个整数化为几个整数的和，有几种表示方法？
dp的主要难点在于需要保证分解的单调性，这样才不会多次计算同一分解的不同排列。通过建立二维dp列表来保证有序性dp[a,b],表示了以b为首位数字的a的划分，并且整个划分必须递减。递减性质由dp的递推来进行保证。
```python
thelist=[]  
while True:  
    try:  
        num=int(input())  
        thelist.append(num)  
    except EOFError:  
        break  
n=51  
#dp[i][j]表示对于i的首位为j的划分种类  
dp=[[0 for _ in range(n+1)] for _ in range(n+1)]  
#初始化  
dp[0][0]=1  
dp[1][1]=1  
#递推  
for i in range(2,n+1):  
    for j in range(1,i+1):  
        dp[i][j]+=sum(dp[i-j][:j+1:])  
for a in thelist:  
    print(sum(dp[a]))
```

**国王游戏**：贪心
国王和大臣们左手右手均有数字。国王打头阵，大臣在后面跟着排队。每个大臣得到奖励数值是前面所有人左手的数字乘积除以它的右手的数字。求出所有排列方式中得到最多奖赏大臣的奖励的最小值。
使用交换论证：交换两个人之间的顺序不改变两个人前面所有人的奖励，也不改变后面的奖励。
如果最大值不出现在这两者之间，那么交换也不改变结果。考虑最大值在两者之间诞生的情况，两种方式分别是max(prid/b1,prid *a1/b2)与max(prid/b2,prid* a2/b1),同时乘以b1 b2/prid,得到max(b2,a1b1)与max(b1,a2b2).若前者大于后者，由于b2< a2b2,a1b1>b1自然成立。那么a1b1>a2b2就变成了上述不等式成立的充分不必要条件。只需将数列按照ab的大小进行排序并计算最大值即可
坑点：由于测试数据较大，向下取整不能使用math.floor(),而应当使用//,否则会出现浮点数精度不够的问题。
```python
n=int(input())  
king=list(map(int,input().split()))  
thelist=[]  
for i in range(n):  
    a,b=map(int,input().split())  
    list_=(a*b,a,b)  
    thelist.append(list_)  
thelist.sort()  
prid=king[0]  
product=0  
for i in range(n):  
    product=max(product,prid//thelist[i][2])  
    prid=prid*thelist[i][1]  
print(product)
```

**并查集**
1.建立一个parent列表表示初始关系，一开始所有i均指向i
2.写两个函数：
	查找函数find：迭代查表。从一个目标出发每次查一步并且把经过的路径上的人全都挂到最后的终点上。如果最终查找到某个人指向自己那就停止查找
	归并函数union：已知某两个人应当属于同一个班级，那么调用find查找两个值的终点，并且要求其中一个终点与另一个挂钩
3.录入初始关系：
	对于已知的关系，调用union.此时由于初始化所有节点指向自己，find第一步就会截止。相当于直接录入初始关系。不用单独写循环
4.遍历n个点，进行find查找并且将查到的“根”录入列表保存。

eg1.学校中班级个数：一共有n个人，m个关系，求出一共有多少个班级
```python
n,m=map(int,input().split())  
#建立关系列表，初始化关系所有节点指向自己  
parent=[i for i in range(n)]  
#路径压缩查找  
def find(x):  
    if parent[x]!=x:  
        parent[x]=find(parent[x])  
    return parent[x]  
#合并不同关系  
def union(x,y):  
    rootx=find(x)  
    rooty=find(y)  
    if rootx!=rooty:  
        parent[rootx]=rooty  
#向关系列表中加入初始关系  
for _ in range(m):  
    a,b=map(int,input().split())  
    union(a-1,b-1)  
  
classlist=[]  
for i in range(n):  
    theclass=find(i)  
    classlist.append(theclass)  
print(len(set(classlist)))
```

**Flood fill算法**：使用queue来进行bfs,帮助快速扩展。用于最大联通区域等问题
```python
from collections import deque  
def bfs(x,y):  
    num=1  
    q=deque([(x,y)])  
    grid[x][y]="."  
    while q:  
        h,k=q.popleft()  
        direction=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]  
        for step in direction:  
            nx=h+step[0]  
            ny=k+step[1]  
            if 0<=nx<=a-1 and 0<=ny<=b-1 and grid[nx][ny]=="W":  
                q.append((nx,ny))  
                grid[nx][ny]="."  
                num+=1  
    return num  
  
n=int(input())  
for _ in range(n):  
    a,b=map(int,input().split())  
    grid=[]  
    for _ in range(a):  
        s=input()  
        s=" ".join(s)  
        list_=s.split()  
        grid.append(list_)  
    #对于grid中的w进行bfs扩展  
    numlist=[0]  
    for i in range(a):  
        for j in range(b):  
            if grid[i][j]=="W":  
                num=bfs(i,j)  
                numlist.append(num)  
    print(max(numlist))
```
**欧拉筛**：
```python
def seive(n):  
    primes=[]#存放已经发现的质数  
    is_prime=[True]*(n+1)  
    for i in range(2,n+1):  
        if is_prime[i]:  
            primes.append(i)  
        for p in primes:  
            if i*p>n:  
                break  
            is_prime[i*p]=False  
            if i%p==0:  #如果i是p的倍数，那么p的下一个素数乘以i的最小因数就不是该素数而是p
                break  
    return primes
```

