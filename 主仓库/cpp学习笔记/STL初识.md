STL大体分为六大组件，分别是:**容器、算法、迭代器、仿函数、适配器（配接器）、空间配置器**
1. 容器：各种数据结构，如vector、list、deque、set、map等,用来存放数据。
2. 算法：各种常用的算法，如sort、find、copy、for_each等
3. 迭代器：扮演了容器与算法之间的胶合剂。
4. 仿函数：行为类似函数，可作为算法的某种策略。
5. 适配器：一种用来修饰容器或者仿函数或迭代器接口的东西。
6. 空间配置器：负责空间的配置与管理。

容器
---
将运用最广泛的一些数据结构实现
常用的数据结构包括**数组,链表,树,栈,队列,集合,映射表**

这些容器分为两种:
	**序列式容器** : 强调值的排序 , 序列式容器中的每个元素均有固定的位置
	**关联式容器** : 二叉树结构 , 各个元素之间没有严格的物理上的顺序关系


算法
---
算法分为*质变算法*和*非质变算法*

质变算法：是指运算过程中会更改区间内的元素的内容。例如拷贝，替换，删除等等

非质变算法：是指运算过程中不会更改区间内的元素内容，例如查找、计数、遍历、寻找极值等等

迭代器
---
提供一种方法，使之能够依序寻访某个容器所含的各个元素，而又无需暴露该容器的内部表示方式。

每个容器都有自己专属的迭代器

迭代器使用非常类似于指针，初学阶段我们可以先理解迭代器为指针

迭代器种类：

| 种类      | 功能                           | 支持运算                        |
| ------- | ---------------------------- | --------------------------- |
| 输入迭代器   | 对数据的只读访问                     | 只读，支持++、==、！=               |
| 输出迭代器   | 对数据的只写访问                     | 只写，支持++                     |
| 前向迭代器   | 读写操作，并能向前推进迭代器               | 读写，支持++、==、！=               |
| 双向迭代器   | 读写操作，并能向前和向后操作               | 读写，支持++、--，                 |
| 随机访问迭代器 | 读写操作，可以以跳跃的方式访问任意数据，功能最强的迭代器 | 读写，支持++、--、[n]、-n、<、<=、>、>= |

常用的容器中迭代器种类为双向迭代器，和随机访问迭代器'

三种遍历方式:
```cpp
	vector<int>::iterator pBegin = v.begin();
	vector<int>::iterator pEnd = v.end();

	//第一种遍历方式：
	while (pBegin != pEnd) {
		cout << *pBegin << endl;
		pBegin++;
	}

	
	//第二种遍历方式：
	for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
		cout << *it << endl;
	}
	cout << endl;

	//第三种遍历方式：
	//使用STL提供标准遍历算法  头文件 algorithm
	for_each(v.begin(), v.end(), MyPrint);
```
# 常用容器

Vector
---
vector.swap 小技巧
```cpp
vector<int>(v).swap(v);
```
是 C++ 中一个经典的内存优化手法，主要用于**收缩 vector 的容量**，释放多余的内存

- `vector<int>(v)`：创建一个**匿名临时 vector 对象**，使用 `v` 的内容进行拷贝构造
- `.swap(v)`：调用匿名对象的 `swap` 方法与原 vector `v` 交换内容
- 最后，匿名对象被销毁，释放内存

```cpp
#include <iostream>
#include <vector>

void demonstrate_swap_trick() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    
    std::cout << "初始状态:" << std::endl;
    std::cout << "size: " << v.size() << ", capacity: " << v.capacity() << std::endl;
    
    // 模拟一些操作导致容量变大
    v.reserve(100);
    std::cout << "After reserve(100):" << std::endl;
    std::cout << "size: " << v.size() << ", capacity: " << v.capacity() << std::endl;
    
    // 应用 swap 技巧
    std::vector<int>(v).swap(v);
    
    std::cout << "After swap trick:" << std::endl;
    std::cout << "size: " << v.size() << ", capacity: " << v.capacity() << std::endl;
}
```

v的capacity很大但稀疏(size小),有大量的闲置内存,
执行之后可以把容量缩小到size的大小


deque
---
double-ended queue双端队列, 可以对头端进行插入删除操作

**deque与vector区别：**

- vector对于头部的插入删除效率低，数据量越大，效率越低
- deque相对而言，对头部的插入删除速度回比vector快
- vector访问元素时的速度会比deque快,这和两者内部实现有关

deque容器的迭代器也是支持随机访问的

### stack

先进后出(first in last out)

### queue

先进后出(first in first out)

### list
链表
链表的组成：链表由一系列**结点**组成

结点的组成：一个是存储数据元素的**数据域**，另一个是存储下一个结点地址的**指针域**

由于链表的存储方式并不是连续的内存空间，因此链表list中的迭代器只支持前移和后移，属于**双向迭代器**

- list容器中不可以通过[]或者at方式访问数据
- 返回第一个元素 --- front
- 返回最后一个元素 --- back

list排序:
```cpp
bool ComparePerson(Person& p1, Person& p2) {

	if (p1.m_Age == p2.m_Age) {
		return p1.m_Height  > p2.m_Height;
	}
	else
	{
		return  p1.m_Age < p2.m_Age;
	}

}
```
- 对于自定义数据类型，必须要指定排序规则，否则编译器不知道如何进行排序
- 高级排序只是在排序规则上再进行一次逻辑规则制定，并不复杂

### set/multiset 

**简介：**

- 所有元素都会在插入时自动被排序

**本质：**

- set/multiset属于**关联式容器**，底层结构是用**二叉树**实现。

**set和multiset区别**：

- set不可以插入重复数据，而multiset可以
- set插入数据的同时会返回插入结果，表示插入是否成功
- multiset不会检测数据，因此可以插入重复数据

set再创建时可以指定自定义的排序方式
```cpp
set<typename , compareWay> s1
```
compareWay是一个类,其中需要重载operator()
```cpp
class MyCompare 
{
public:
	bool operator()(int v1, int v2) {
		return v1 > v2;
	}
};
```

### map / multimap

**简介：**

- map中所有元素都是pair
- pair中第一个元素为key（键值），起到索引作用，第二个元素为value（实值）
- 所有元素都会根据元素的键值自动排序

**本质：**

- map/multimap属于**关联式容器**，底层结构是用二叉树实现。

**优点：**

- 可以根据key值快速找到value值

map和multimap**区别**：

- map不允许容器中有重复key值元素
- multimap允许容器中有重复key值元素

## STL 函数对象
**概念：**

- 重载**函数调用操作符**的类，其对象常称为**函数对象**
- **函数对象**使用重载的()时，行为类似函数调用，也叫**仿函数**

**本质：**

函数对象(仿函数)是一个**类**，不是一个函数

**特点：**

- 函数对象在使用时，可以像普通函数那样调用, 可以有参数，可以有返回值
- 函数对象超出普通函数的概念，函数对象可以有自己的状态
- 函数对象可以作为参数传递

### 谓词

**概念：**

- 返回bool类型的仿函数称为**谓词**
- 如果operator()接受一个参数，那么叫做一元谓词
```cpp
	struct GreaterFive{
		bool operator()(int val) {
			return val > 5;
		}
	};
	vector<int> v;
	vector<int>::iterator it = find_if(v.begin(), v.end(),GreaterFive());
  ```
- 如果operator()接受两个参数，那么叫做二元谓词
```cpp
class MyCompare
{
public:
	bool operator()(int num1, int num2)
	{
		return num1 > num2;
	}
};

vector<int> v;
//默认从小到大
sort(v.begin(), v.end());
//使用函数对象改变算法策略，排序从大到小
sort(v.begin(), v.end(), MyCompare());
```

## STL- 常用算法

**概述**:

- 算法主要是由头文件`<algorithm>` `<functional>` `<numeric>`组成。
- `<algorithm>`是所有STL头文件中最大的一个，范围涉及到比较、 交换、查找、遍历操作、复制、修改等等
- `<numeric>`体积很小，只包括几个在序列上面进行简单数学运算的模板函数
- `<functional>`定义了一些模板类,用以声明函数对象。
### 5.1 常用遍历算法
- `for_each(iterator beg, iterator end, _func);` 
    实现遍历容器
    
    // 遍历算法 遍历容器元素
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // _func 函数或者函数对象

- - `transform(iterator beg1, iterator end1, iterator beg2, _func);`
	搬运容器到另一个容器中
	注意,在搬运之前需要把目标容器resize到足够大小
	
	//beg1 源容器开始迭代器
	
	//end1 源容器结束迭代器
	
	//beg2 目标容器开始迭代器
	
	//_func 函数或者函数对象

### 5.2 常用查找算法

- `find` //查找元素
	`find(iterator beg, iterator end, value);`
	// 按值查找元素，找到返回指定位置迭代器，有多个目标元素会返回第一个的迭代器，找不到返回结束迭代器位置
	
	// beg 开始迭代器
	
	// end 结束迭代器
	
	// value 查找的元素
	
- `find_if` //按条件查找元素
	`find_if(iterator beg, iterator end, _Pred);` 
    
    // 按值查找元素，找到返回指定位置迭代器，找不到返回结束迭代器位置
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // \_Pred 函数或者谓词（返回bool类型的仿函数）
    
- `adjacent_find` //查找相邻重复元素
	`adjacent_find(iterator beg, iterator end);` 
	
	// 查找相邻重复元素,返回相邻元素的第一个位置的迭代器
	
	// beg 开始迭代器
	
	// end 结束迭代器
	
- `binary_search` //二分查找法
	`bool binary_search(iterator beg, iterator end, value);` 
    
    // 查找指定的元素，查到 返回true 否则false
    
    // 注意: 在****无序序列中不可用***
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // value 查找的元素
    
- `count` //统计元素个数
	`count(iterator beg, iterator end, value);` 
    
    // 统计元素出现次数
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // value 统计的元素
    
- `count_if` //按条件统计元素个数
	`count_if(iterator beg, iterator end, _Pred);` 
    
    // 按条件统计元素出现次数
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // _Pred 谓词
    
    ​

### 5.3 常用排序算法

- `sort` //对容器内元素进行排序
	`sort(iterator beg, iterator end, _Pred);` 
    
    // 按值查找元素，找到返回指定位置迭代器，找不到返回结束迭代器位置
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // _Pred 谓词
    
- `random_shuffle` //洗牌 指定范围内的元素随机调整次序
	`random_shuffle(iterator beg, iterator end);` 
    
    // 指定范围内的元素随机调整次序
    
    // beg 开始迭代器
    
    // end 结束迭代器
    ​
- `merge`  // 容器元素合并，并存储到另一容器中
	`merge(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);` 
	
	// 容器元素合并，并存储到另一容器中
	
	// 注意: 两个容器必须是**有序的** , merge使用的[[排序|归并排序]] , 如果两个容器无序,则merge无意义
	
	// beg1 容器1开始迭代器  
	// end1 容器1结束迭代器  
	// beg2 容器2开始迭代器  
	// end2 容器2结束迭代器  
	// dest 目标容器开始迭代器
	
- `reverse` // 反转指定范围的元素
	`reverse(iterator beg, iterator end);` 
	
	// 反转指定范围的元素
	
	// beg 开始迭代器
	
	// end 结束迭代器

### 5.4 常用拷贝和替换算法

- `copy` // 容器内指定范围的元素拷贝到另一容器中
	`copy(iterator beg, iterator end, iterator dest);` 
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // dest 目标起始迭代器
    
- `replace` // 将容器内指定范围的旧元素修改为新元素
	`replace(iterator beg, iterator end, oldvalue, newvalue);` 
    
    // 将区间内旧元素 替换成 新元素
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // oldvalue 旧元素
    
    // newvalue 新元素
- `replace_if`  // 容器内指定范围满足条件的元素替换为新元素
	`replace_if(iterator beg, iterator end, _pred, newvalue);` 
    
    // 按条件替换元素，满足条件的替换成指定元素
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // _pred 谓词
    
    // newvalue 替换的新元素
- `swap` // 互换两个容器的元素
	`swap(container c1, container c2);` 
	
	// 互换两个容器的元素
	
	// c1容器1
	
	// c2容器2

### 5.5 常用算术生成算法
- `accumulate` // 计算容器元素累计总和
    `accumulate(iterator beg, iterator end, value);` 
    
    // 计算容器元素累计总和
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // value 起始值
- `fill` // 向容器中添加元素
	`fill(iterator beg, iterator end, value);` 
    
    // 向容器中填充元素
    
    // beg 开始迭代器
    
    // end 结束迭代器
    
    // value 填充的值

### 5.6 常用集合算法
- `set_intersection` // 求两个容器的交集
    `set_intersection(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);` 
    
    // 求两个集合的交集
    
    // **注意:两个集合必须是有序序列**
    
    // beg1 容器1开始迭代器  
    // end1 容器1结束迭代器  
    // beg2 容器2开始迭代器  
    // end2 容器2结束迭代器  
    // dest 目标容器开始迭代器
    
    求交集的两个集合必须的有序序列
	目标容器开辟空间需要从**两个容器中取小值**
	set_intersection返回值既是交集中最后一个元素的位置
	
- `set_union` // 求两个容器的并集
    `set_union(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);` 
    
    // 求两个集合的并集
    
    // **注意:两个集合必须是有序序列**
    
    // beg1 容器1开始迭代器  
    // end1 容器1结束迭代器  
    // beg2 容器2开始迭代器  
    // end2 容器2结束迭代器  
    // dest 目标容器开始迭代器
    
    求并集的两个集合必须的有序序列
	目标容器开辟空间需要**两个容器相加**
	set_union返回值既是并集中最后一个元素的位置​
	
- `set_difference`  // 求两个容器的差集
	`set_difference(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);` 
	
	// 求两个集合的差集
	
	// **注意:两个集合必须是有序序列**
	
	// beg1 容器1开始迭代器  
	// end1 容器1结束迭代器  
	// beg2 容器2开始迭代器  
	// end2 容器2结束迭代器  
	// dest 目标容器开始迭代器
	
	求差集的两个集合必须的有序序列
	目标容器开辟空间需要从**两个容器取较大值**
	set_difference返回值既是差集中最后一个元素的位置