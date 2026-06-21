## 注释:
\-\-是单行注释
\-\-\[\[ 是多行注释, 结尾是
\-\-\]\]
\[ 和 \[ 中间加入等号可以作为注释分级标注, 只需要在\] 和 \]中间加入对应数量的等号即可闭合
如
```lua
--[[
	这是0级注释
	--[=[
	这是1级注释
	--[==[
	这是二级注释
]===]
]=]
]]
```

## 全局变量
默认情况下,变量总是认为是全局的, 不需要声明, 给一个变量赋值后即创建, 访问未初始化的变量不会报错, 会返回nil

## 数据类型
8个数据类型为nil , boolean , number , string , userdata, function, thread, table 

其中
nil  表示空, 无有效值, nil作比较时要加上引号
```lua
type(x)=='nil'
type(type(x))==string
```
boolean 只有两个可选值, false 和 nil 都是false , 其他都为true(包括数字0也是true)
number表示双精度类型的实浮点数
string表示的字符串可以由一堆双引号或单引号表示
	可以用2个方括号"[[]]"来表示 , "一块"字符串(内部可以回车)
	在对一个数字字符串上进行算数操作(+-\*/)时, lua会尝试将这个数字字符串转成一个数字
	数字和字符串混合拼接时, 数字会自动转化成字符串,  `..` 作为字符串拼接的操作符, `+` 仅用于数字相加, 不能用于字符串拼接
	使用 # 来计算字符串的长度，放在字符串前面```len="hello"
	print(#len)
	> 5
	```
funcition是由C或lua编写的函数
	函数是被看作是"第一类值（First-Class Value）"，函数可以存在变量里
	可以以匿名函数（anonymous function）的方式通过参数传递
	```
	-- function_test2.lua 脚本文件
	function testFun(tab,fun)
        for k ,v in pairs(tab) do
            print(fun(k,v));
        end
	end
	tab={key1="val1",key2="val2"};
	testFun(tab,
	function(key,val)--匿名函数
	    return key.."="..val;
	end
	);
	```
userdata表示任意存储在变量中的C数据结构
	userdata 是一种用户自定义数据，用于表示一种由应用程序或 C/C++ 语言库所创建的类型，可以将任意 C/C++ 的任意数据类型的数据（通常是 struct 和 指针）存储到 Lua 变量中调用。
thread 表示执行的独立线路, 用于执行协同程序
	在 Lua 里，最主要的线程是协同程序（coroutine）。它跟线程（thread）差不多，拥有自己独立的栈、局部变量和指令指针，可以跟其他协同程序共享全局变量和其他大部分东西。
	线程跟协程的区别：线程可以同时多个运行，而协程任意时刻只能运行一个，并且处于运行状态的协程只有被挂起（suspend）时才会暂停。
table Lua 中的表（table）其实是一个"关联数组"（associative arrays），数组的索引可以是数字、字符串或表类型。在 Lua 里，table 的创建是通过"构造表达式"来完成，最简单构造表达式是{}，用来创建一个空表。
	使用{}来初始化一个表`local tbl2 = {"apple", "pear", "orange", "grape"}`
	数组的索引可以使数字或者字符串
	```
	a={}
	a["key"]="value"
	key = 10
	a[key]=22
	a[key] = a[key] + 11
	for k, v in pairs(a) do 
		print(k .. ":" .. v)
	end
	---------
	key : value
	10 : 33
	```
	lua里表一般以1开始
	table大小不固定, 有新数据添加时table长度会自动增长
