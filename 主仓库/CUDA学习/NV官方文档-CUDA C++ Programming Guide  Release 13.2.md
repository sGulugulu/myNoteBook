
线程块组(cluster) :  (计算能力9.0以上)
	由线程块组成,类似于线程块中的线程,保证在流式多处理器上协同调度,集群中的线程块也保证在gpu的gpu处理集群(GPC)上协同调度,
	*支持自定义为一维,二维,三维,集群最多支持8个线程块* 
	调用: 使用`__cluster_dims__((X,Y,Z))`或通过cuda内核启动API`cudaLaunchKernelEx`在内核中启用
如果用__block_size__指定cluster大小,那么需要传入2个三元组参数,第一个表示块维度,第二个表示簇维度.

当使用blockDim定义kernel函数(blockDim固定了block维度和kernel的cluster维度大小),

此时调用kernel函数就*无须再传入blockdim*,只写griddim和smem,stream即可(也可以省)

另外,\_\_block_size\_\_的第二个元组和\_\_cluster_dims\_\_  不能同时指定
	它们都是指定cluster簇维度大小的,会起冲突
	当指定\_\_block_size\_\_的第二组元组时,表示为"块作为簇"功能一起用,编译期会将<<<>>>内的第一个参数识别为cruatal而非block的数量
	