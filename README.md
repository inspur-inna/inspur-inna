![Image text](https://github.com/inspur-inna/inspur-inna/blob/master/Image/inspur.png)

# 基于FPGA的CNN自适应映射技术——inspur-inna

基于FPGA板卡设计深度学习加速器并进行优化，在整体性能和功耗方面拟达到业界领先水平，映射技术采用宏指令的Look-Aside Acceleration框架，实现了一键式快速部署、软硬件协同优化、支持多种卷积、执行过程无需主机干预。本项目为映射技术的软件端，拟实现CNN映射编译器和CNN量化器，首先由TensorFlow产生的模型文件解析产生CNN的计算图模型，CNN映射编译器会根据解析的计算图和现有的CNN加速库单元，选择相应的CNN库单元，生成相应的硬件结构和相应的调度器的配置参数，以达到计算、片上存储、片上带宽和片外带宽的均衡，从而达到最优的计算性能；CNN量化器可根据模型的权重文件，对各层数据进行8位定点量化，以便于FPGA的DSP计算，从而在保证精度的前提下降低存储开销，提高处理速度，降低功耗。


## Install

### inna install
TVM need LLVM，LLVM install in Ubuntu（other system require source code compilation）
```bash
apt search llvm
apt install llvm-6.0
apt install clang-6.0
```

Install miniconda for python=3.6，install_inna.sh include TVM install script（refer to TVM <https://tvm.apache.org/docs/install/from_source.html>）
```bash
conda create -n inna python=3.6 ipykernel -y
conda activate inna
cd inna/tools && ./install_inna.sh
```


## Run

### Compiler

Compiler  in TensorFlow or Mxnet or keras or onnx.
```bash
$ python compiler.py
```

### Quantizer

Quantizer  in TensorFlow.
```bash
$ python quantize.py
```

### Runtime

Runtime  in ours.
```bash
$ python runtime.py
```

## Open source disclaimer
 【0416】inna开源代码免责声明.docx
