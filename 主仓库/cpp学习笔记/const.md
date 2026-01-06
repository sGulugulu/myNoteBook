  
本质：const在谁后面谁就不可修改，const在最前面则将其后移一位即可，二者等效

const type \*a  和 type const \*a:
	Bjarne 在他的 The C++ Programming Language 里面给出过一个助记的方法： **把一个声明从右向左读**
```cpp
char  * const cp; ( * 读成 pointer to ) 
cp is a const pointer to char 

const char * p; 
p is a pointer to const char; 

char const * p;
```
同上因为 C++ 里面没有 const* 的运算符，所以 const 只能属于前面的类型。

**修饰指针**

修饰指针的情况比较多，主要有以下几种情况：

1、const 修饰 \***p**，指向的对象只读，指针的指向可变：
```cpp
int a = 9;
int b = 10;
const int *p = &a;//p是一个指向int类型的const值,与int const *p等价
*p = 11;    //编译错误，指向的对象是只读的，不可通过p进行改变
p = &b;     //合法，改变了p的指向
```
这里为了便于理解，可认为const修饰的是 *p，通常使用 ***** 对指针进行解引用来访问对象，因而，该对象是只读的。

2、const 修饰 p，指向的对象可变，指针的指向不可变：
```cpp
int a = 9;
int b = 10;
int * const p = &a;//p是一个const指针
*p = 11;    //合法，
p = &b;     //编译错误，p是一个const指针，只读，不可变
```
3、指针不可改变指向，指向的内容也不可变
```cpp
int a = 9;
int b = 10;
const int * const p = &a;//p既是一个const指针，同时也指向了int类型的const值
*p = 11;    //编译错误，指向的对象是只读的，不可通过p进行改变
p = &b;     //编译错误，p是一个const指针，只读，不可变
```

看完上面几种情况之后是否会觉得混乱，并且难以记忆呢？我们使用一句话总结：

const 放在 * 的左侧任意位置，限定了该指针指向的对象是只读的；const放在 * 的右侧，限定了指针本身是只读的，即不可变的。

如果还不是很好理解，我们可以这样来看，去掉类型说明符，查看 const修饰的内容，上面三种情况去掉类型说明符 int 之后，如下：
```cpp
const *p; //修饰*p，指针指向的对象不可变
* const p; //修饰p，指针不可变
const * const p; //第一个修饰了*p，第二个修饰了p，两者都不可变
```
const 右边修饰谁，就说明谁是不可变的。上面的说法仅仅是帮助理解和记忆。借助上面这种理解，就会发现以下几种等价情况：
```cpp
const int NUM = 10; //与int const NUM等价
int a = 9;
const int *p  = &a;//与int const *p等价
const int arr[] = {0,0,2,3,4}; //与int const arr[]等价
```
