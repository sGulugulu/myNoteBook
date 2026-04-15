Nsight Systems:综合的,系统级的调优工具,用于协同cpu,gpu的工作负载,从pipeline流程图中看出瓶颈所在

Nsight Compute:针对kernel核函数的性能进行分析,包括tensor core上跑的指令集,memory使用情况,硬件sm使用情况的详细性能分析文件

Nsight Graphics:图形学渲染相关

优化流程: profiling ->分析结果 -> 优化 ->重新profiling

## Nsight system
- 系统级的调优,可以捕捉到gpu和cpu的事件
- 可以快速定位优化的机会
- 可以把任务均匀地分配在cpu和gpu之间
- 全平台,支持Lunx ,win,mac

计算: 支持 cuda api ,kernel launch , 执行关联
	方便后期分析瓶颈
库: cuBLAS ,cuDNN , TensorRT
图形 : vulkan ,opengl ,dx11/12 , DXR ,V-sync
操作系统线程状态, CPU优化情况 , pthread , file I/O ,NVTX(一个库,用于打标分析独立事件)

### cpu thread
黑色: cpu利用率
灰色: cpu在等待
有颜色 : cpu在活动
不同颜色 : 不同的cpucore
下面: cuda api调用的kernel名字
还会捕捉调用的栈,便于找到等待的原因


### cuda api
可以看到kernel是何时由cpu启动的
kernel启动的开销

gpu工作负载
![[Pasted image 20260415164226.png]]
蓝色是kernel coverage的平均情况,较高代表这段时间内一致在运行kernel,
红色是memory的平均操作数
还可以高亮cpu的kernel调用到gpu的kernel执行,分析瓶颈
### 支持NVTX
对kernel代码添加nvtx的标签,timeline中会显示标签,方便调优
cpp
`#include "nvToolsExt.h" `
同时支持python Tensor
```cpp
include "nvToolsExt.h"
...
void myfunction( int n, double * x ){
	nvtxRangePushA("init host data");
	//initialize x on host
	init_host_data(n,x,x_d,y_d) ;
	nvtxRangePop () ;
	}
```

## nsight compute
交互式的kernel,主要用于profiling , 可视化gpu上的speed of light(理论上线),基于baseline的比较功能,方便优化后对比

