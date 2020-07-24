
<img src="https://https://github.com/inspur-inna/inspur-inna/tree/master/Image/inspur.png" width="100"></a>
# 基于FPGA的CNN自适应映射技术---inspur-inna

基于宏指令的Look-Aside Acceleration框架：

- 一键式快速部署
- 软硬件协同优化
- 支持多种卷积
- 执行过程无需主机干预



## Install

### TVM source code install
LLVM install in Ubuntu
```bash
apt search llvm
apt install llvm-6.0
apt install clang-6.0
```

TVM Install Source<https://tvm.apache.org/docs/install/from_source.html>

### inna install
Install miniconda for python=3.6
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
