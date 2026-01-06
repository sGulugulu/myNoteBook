## nv 30系gpu L1 cache ,shared memory ,global memory ,dram之间的读取速率对比:

| 内存层级            | 物理位置             | 延迟 (时钟周期)   | 理论/实际带宽         | 容量范围                 | 主要用途与说明                                                             |
| --------------- | ---------------- | ----------- | --------------- | -------------------- | ------------------------------------------------------------------- |
| **L1缓存 / 共享内存** | **SM内部** (片上)    | **1~32**    | **极高** (数TB/s级) | **每SM 128KB**        |                                                                     |
| **L2缓存**        | GPU芯片上 (共享)      | 32~64       | 高               | **全体SM共享 2MB - 6MB** | **所有访问全局内存的请求都经过它**，是GPU芯片上最后一级缓存。                                  |
| **全局内存 (DRAM)** | **GPU板载显存** (片外) | **400~600** | **高 (数百GB/s)**  | **数GB - 24GB**       | **所有线程可访问**，容量大但延迟高。实测带宽因型号而异（如RTX 3080约669 GB/s，RTX 3070约388 GB/s） |

这些层级构成了GPU的内存访问路径。当SM中的线程需要数据时，硬件会按以下顺序查找，而速度差距主要由**物理距离**决定：

1. **最快路径（1-32周期）**：先在**寄存器**和**L1/共享内存**（在SM内部）中查找。
    
2. **次级路径（32-64周期）**：若未命中，则查询所有SM共享的**L2缓存。
    
3. **最慢路径（400-600周期）**：如果L2也缺失，就必须访问位于GPU芯片外的**全局内存(DRAM)**，这也是延迟最高的环节。



## 为什么L1缓存的时钟周期是32

**它不是一个固定延迟，而是一个受多种因素影响的范围，而32通常是一个关键的设计上限或典型值。**

### 🔍 核心原因：这并非单一延迟，而是一个“延迟范围”

像NVIDIA 30系（Ampere架构）这样的现代GPU，其L1缓存访问延迟不是固定的。**1~32周期**这个范围，实际上描述了从**最理想情况**到**最复杂情况**的全过程。延迟的变化主要源于两个根本机制：

1. **“命中”与“未命中”的根本路径差异**
    
    - **理想情况（低延迟，如几个周期）**：当线程需要的数据**正好在L1缓存中**（称为“命中”），SM可以直接从身边的存储阵列中取出数据，速度极快。
        
    - **复杂情况（高延迟，接近32周期）**：当数据**不在L1缓存中**（称为“未命中”），硬件就必须发起一个更复杂的请求。这个请求可能需要经过共享内存控制器，甚至继续前往L2缓存。每一次转发和寻址都会增加延迟。
        
2. **共享内存的访问复杂性**  
    在GPU中，L1缓存和共享内存**物理上是同一块高速的SRAM存储体**。这意味着：
    
    - 当它被**配置为L1缓存**时，其操作对程序员是透明的，由硬件自动管理数据存放位置。
        
    - 当它被**作为共享内存**显式使用时，其访问延迟就与程序员的代码模式直接相关。最典型的问题是 **“Bank冲突”**——如果同一个线程束（Warp）中的多个线程同时访问同一个存储体（Bank）的不同地址，这些访问就必须串行化，从而将延迟从最优的1-2个周期拉长到32个周期（最坏情况下，32个线程依次排队访问）。
        
### 🧩 “32”这个数字的由来

所以，“32”这个周期数，可以理解为GPU设计者为**处理一次完整、复杂的内存请求所预留的典型时间窗口**，特别是在涉及**缓存未命中**或**共享内存Bank冲突**的场景下。在这个时间窗口内，SM的线程调度器会做一件至关重要的事：**切换执行其他就绪的线程束（Warp）**。

这是GPU实现高性能并行的核心秘籍。当某个线程束在等待内存数据（比如这32个周期）时，SM可以立即切换到另一个不需要等待数据的线程束去执行计算，从而将内存延迟“隐藏”在计算之下，保持计算单元的持续忙碌。

理解:
32个周期是在最差情况下的,一个warp有32个thread , 当出现他们都要访问同一个bank内不同地址的数据[1],会出现bank 冲突, 为了避免bank冲突, 硬件会把这32个threads 排好队,轮流访问, 造成32周期的延迟

[1]:(对于某些特定的访问模式（如一个Warp内所有线程读取**完全相同的地址**，即“广播”），硬件会进行优化，不会产生32周期延迟。冲突延迟更多地发生在访问**同一Bank内不同地址**的模式下。)


## L1缓存和shared memory什么区别和关系

L1缓存和共享内存的关系是“**一体两面**”。它们物理上是**同一块硬件存储**，但在逻辑上被划分为两种不同功能的内存，供程序员以不同方式使用。

简单来说，你可以把这块存储体想象成一个 **“可变形”的资源池**。它的总容量固定（例如每SM 128KB），但可以根据计算需求，通过软件配置改变L1缓存和共享内存的容量分配。

|特性维度|**L1缓存**|**共享内存**|
|---|---|---|
|**本质**|**硬件自动管理的缓存**|**软件显式管理的可编程内存**|
|**管理方式**|完全由硬件控制。硬件决定缓存什么数据、何时替换。对程序员**透明**。|完全由程序员通过代码控制。需要手动声明、加载数据、控制访问。|
|**访问方式**|**隐式访问**。内核代码读写全局/常量内存时，自动经由L1缓存。|**显式访问**。内核代码必须使用`__shared__`变量或API来读写。|
|**数据生命周期**|由硬件替换算法（如LRU）决定，与数据是否被使用有关。|与**线程块**绑定。线程块开始时分配，块内所有线程可见，块结束时释放。|
|**性能关键**|**局部性**。应优化程序的空间/时间局部性以提高命中率。|**访问模式**。必须精心设计以避免**Bank冲突**和保证**合并访问**。|
|**主要用途**|加速对全局内存、常量内存等**片外内存**的重复访问。|作为**线程块内部的高速暂存器**，用于**线程间通信**、**数据重用**、**全局内存访问的中间站**。|
|**类比**|**大脑的短期记忆**。你无意识地记住最近看到或想到的东西。|**团队工作时的共享白板**。队员（线程）主动把需要讨论的信息写上去、读出来。|

## 详细说明<<<>>>中的参数,以及这些参数可能的类型

### **参数类型总结表**

|参数|作用|允许的类型|默认值|限制|
|---|---|---|---|---|
|**Dg (网格)**|定义线程块布局|`int`, `dim3`, 表达式|无|`gridDim.x ≤ 2³¹-1`  <br>`gridDim.y ≤ 65535`  <br>`gridDim.z ≤ 65535`|
|**Db (线程块)**|定义每个块的线程|`int`, `dim3`, 常量, 变量|无|`blockDim.x ≤ 1024`  <br>`blockDim.y ≤ 1024`  <br>`blockDim.z ≤ 64`  <br>`总线程数 ≤ 1024`|
|**Ns (共享内存)**|动态共享内存大小|`size_t`, `int`, 表达式|0|`≤ sharedMemPerBlock`  <br>（通常48KB或96KB）|
|**S (流)**|执行流|`cudaStream_t`, `0`|`0`|必须是有效流或0|
### **完整语法格式**

```cpp
kernel<<<Dg, Db, Ns, S>>>(argument_list);
```

四个参数分别是：

- **`Dg`** - 网格维度 (Grid Dimension)---线程块的数量和布局
	 - **可能的类型**
		- 整数 `kernel<<<100,100>>>`
		- dim3 结构体 , 创建多维网格 
			```cpp
			dim3 grid(10, 20);                // 10×20个线程块 (二维)
			kernel<<<grid, ...>>>();
			
			dim3 grid3(10, 20, 5);            // 10×20×5个线程块 (三维)
			kernel<<<grid3, ...>>>();
			
			kernel<<<dim3(1,2,3),2,...>>>();   //可以整数和dim3组合,也可以在<<<>>>中构造
			```
		- 表达式
			```cpp
			int N = 1000;
			int threads_per_block = 256;
			kernel<<<(N + threads_per_block - 1) / threads_per_block, ...>>>();
			```
	- 实际限制(通过cudaGetDeviceProperties获取）
- **`Db`** - 线程块维度 (Block Dimension)
	- 可能的类型
		```cpp
	  // 1. 整数 - 创建一维线程块
		kernel<<<..., 256>>>();           // 每个线程块256个线程，x方向
		
		// 2. dim3结构体 - 创建多维线程块
		dim3 block(16, 16);               // 16×16=256个线程 (二维)
		kernel<<<..., block>>>();
		
		dim3 block3(8, 8, 4);             // 8×8×4=256个线程 (三维)
		kernel<<<..., block3>>>();
		
		// 3. 常量表达式
		constexpr int BLOCK_SIZE = 256;
		kernel<<<..., BLOCK_SIZE>>>();
		
		// 4. 变量（必须在编译时已知或运行时指定）
		int block_size = 256;
		kernel<<<..., block_size>>>();    // 运行时指定
		  ```
	- 维度限制(`prop.maxThreadsPerBlock`)
	- 实践大小(32的倍数(warp的大小))
- **`Ns`** - 共享内存大小 (Shared Memory Size)
	- 基本用法
		```cpp
		// 0 或 不指定：使用默认共享内存（无动态共享内存）
		kernel<<<grid, block>>>();               // 默认0
		kernel<<<grid, block, 0>>>();            // 显式指定0
		
		// 正整数：为每个线程块分配的共享内存字节数
		size_t shared_mem_size = 1024;           // 1KB共享内存
		kernel<<<grid, block, shared_mem_size>>>();
		
		// 使用sizeof计算大小
		float shared_array[256];
		size_t size = sizeof(shared_array);      // 256 * 4 = 1024字节
		kernel<<<grid, block, size>>>();
		
		// 动态计算大小
		int elements_per_block = 128;
		size_t shmem = elements_per_block * sizeof(float);
		kernel<<<grid, block, shmem>>>();
		```
	- 在核函数中使用动态共享内存
		```cpp
		// 1. 声明外部共享内存
		__global__ void kernel(float* data) {
		    extern __shared__ float s_data[];
		    
		    // s_data的大小由启动配置中的Ns参数决定
		    int tid = threadIdx.x;
		    s_data[tid] = data[tid];
		    __syncthreads();
		    
		    // 处理共享内存数据...
		}
		
		// 2. 启动时指定大小
		float* d_data;
		size_t shmem_size = block_size * sizeof(float);
		kernel<<<grid_size, block_size, shmem_size>>>(d_data);
		  ```
	- 混合静态和动态共享内存
		```cpp
		// 核函数同时使用静态和动态共享内存
		__global__ void kernel(float* data) {
		    // 静态共享内存（编译时确定大小）
		    __shared__ float static_smem[32];
		    
		    // 动态共享内存（运行时指定大小）
		    extern __shared__ float dynamic_smem[];
		    
		    // 注意：静态和动态共享内存连续分配
		    // 总共享内存 = 静态大小 + 动态大小
		}
		
		// 启动时只需指定动态部分的大小
		kernel<<<grid, block, dynamic_size>>>(data);
		```
	- 共享内存限制
		```cpp
		// 获取共享内存限制
		printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
		// 典型值：48KB 或 96KB
		
		// 计算共享内存使用
		size_t required_shmem = ...;
		if (required_shmem > prop.sharedMemPerBlock) {
		    printf("Error: Shared memory exceeds limit!\n");
		}
		
		// 调整配置
		int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
		// 共享内存大小影响每个SM能容纳的线程块数
		```
- **`S`** - CUDA流 (Stream)
	- 基本用法
		```cpp
		// 0 或 不指定：使用默认流（同步流）
		kernel<<<grid, block>>>();               // 默认流
		kernel<<<grid, block, 0, 0>>>();         // 显式指定默认流
		
		// cudaStream_t：指定特定的流
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		
		kernel<<<grid, block, 0, stream>>>();    // 在指定流中执行
		
		cudaStreamDestroy(stream);               // 销毁流
		```
	- 流的类型
		```cpp
		// 1. 默认流 (NULL stream / 0)
		//    阻塞流，所有操作顺序执行
		kernel<<<grid, block, 0, 0>>>();
		
		// 2. 创建非默认流
		cudaStream_t stream1, stream2;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
		
		// 3. 使用多个流实现并行
		kernel1<<<grid1, block1, 0, stream1>>>(...);
		kernel2<<<grid2, block2, 0, stream2>>>(...);
		// 两个核函数可能并发执行（如果硬件支持）
		
		// 4. 高级流选项
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		cudaStreamCreateWithPriority(&stream, cudaStreamDefault, priority);
		```
	- 流同步
		```cpp
		// 1. 默认流 (NULL stream / 0)
		//    阻塞流，所有操作顺序执行
		kernel<<<grid, block, 0, 0>>>();
		
		// 2. 创建非默认流
		cudaStream_t stream1, stream2;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
		
		// 3. 使用多个流实现并行
		kernel1<<<grid1, block1, 0, stream1>>>(...);
		kernel2<<<grid2, block2, 0, stream2>>>(...);
		// 两个核函数可能并发执行（如果硬件支持）
		
		// 4. 高级流选项
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		cudaStreamCreateWithPriority(&stream, cudaStreamDefault, priority);
		```
