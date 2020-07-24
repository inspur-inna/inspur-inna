#!/bin/bash

# install python dependecies
pip install -r ../requirements.txt

git clone https://github.com/apache/incubator-tvm.git --recurse-submodules
cd tvm
git pull origin --recurse-submodules
mkdir build
cp cmake/config.cmake build
cd build
cmake3 ..
make -j24
cd ..

cd python; python setup.py install; cd ..
cd topi/python; python setup.py install; cd ../..
#cd nnvm/python; python setup.py install; cd ../..

cd ..

# install inna
pip install ..
