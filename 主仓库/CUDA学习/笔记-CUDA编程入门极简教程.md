
[CUDA编程入门极简教程 - 知乎](https://zhuanlan.zhihu.com/p/34587739)

## 概念

host: 代之cpu及其内存
device : 指代GPU及其内存
CUDA既包含host程序,又包含device程序,它们分别在cpu和gpu上运行,它们之间可以进行通信,进行数据拷贝

kernel: 是在device上线程中并行执行的函数,核函数用__global__符号声明,在调用时需要用`<<<grid,block>>>`来指定kernel要执行的线程数量,在CUDA中,每个线程都要执行核函数,并且每个线程会分配一个唯一的线程好thread ID , 这个ID值可以通过核函数的内置变量 `threadIdx`来获得

由于GPU实际上是异构模型,所以需要区分host和device上的代码，在CUDA中是通过函数类型限定词开区别host和device上的函数，主要的三个函数类型限定词如下：

- `__global__`：在device上执行，从host中调用（一些特定的GPU也可以从device上调用），返回类型必须是`void`，不支持可变参数参数，不能成为类成员函数。注意用`__global__`定义的kernel是异步的，这意味着host不会等待kernel执行完就执行下一步。
- `__device__`：在device上执行，单仅可以从device中调用，不可以和`__global__`同时用。只能被其他`__device`函数或者`__global__`函数调用
- `__host__`：在host上执行，仅可以从host上调用，一般省略不写，不可以和`__global__`同时用，但可和`__device__`，此时函数会在device和host都编译。

典型的CUDA程序的执行流程如下:
1. 分配host内存，并进行数据初始化；
2. 分配device内存，并从host将数据拷贝到device上；
3. 调用CUDA的核函数在device上完成指定的运算；
4. 将device上的运算结果拷贝到host上；
5. 释放device和host上分配的内存。

用__global__修饰的函数是核函数
定义:
```cpp
__global__ void add(int a, int b, int *c) { 
    *c = a + b; 
}
```
- `__global__` 修饰符表明这是GPU核函数
    
- 参数 `a`, `b` 通过值传递（复制到每个线程）
    
- 参数 `c` 通过指针传递（指向GPU显存）
    
- 只有一个线程在执行这个加法操作
调用方法:
```cpp
add<<<1,1>>> (2,7,dev_c);
```

```cpp
kernel<<<Dg, Db, Ns, S>>>(argument_list);
```
四个参数分别是：

1. **`Dg`** - 网格维度 (Grid Dimension)
    
2. **`Db`** - 线程块维度 (Block Dimension)
    
3. **`Ns`** - 共享内存大小 (Shared Memory Size)
    
4. **`S`** - CUDA流 (Stream)



void*  cudaMalloc(void** ptr, size_t size)函数:

- 第一个参数是一个指针，这个指针指向你希望保存新分配内存地址的那个指针变量。
- 第二个参数是要分配的内存大小

cudaMalloc 分配的是device内存,只能在GPU代码中使用
因此它返回的指针不能在host中解引用
- 可以将使用 cudaMalloc() 分配的指针传递给在设备上执行的函数。
- 可以使用 cudaMalloc() 分配的指针从设备上执行的代码读取或写入内存。
- 可以将使用 cudaMalloc() 分配的指针传递给在主机上执行的函数。
- 不能使用 cudaMalloc() 分配的指针从主机上执行的代码读取或写入内存。

记住这个简单的规则：**设备指针在设备代码中是"真实的指针"，在主机代码中是"句柄"**。

- ✅ **在设备上**：可以读写设备指针指向的内存
    
- ✅ **在主机上**：可以传递、存储设备指针，但不能解引用
    
- 🔄 **数据传输**：必须使用 `cudaMemcpy` 在主机和设备间复制数据

>`cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);`
>dst：目标指针
>src：源指针
>count：复制的字节数
>kind：复制方向，常见选项：
>	cudaMemcpyHostToDevice：主机 → 设备
>	cudaMemcpyDeviceToHost：设备 → 主机
>	cudaMemcpyDeviceToDevice：设备内部复制
>如果源指针和目标指针都在主机上，我们只需使用标准 C 的 memcpy（）例程在它们之间复制。

