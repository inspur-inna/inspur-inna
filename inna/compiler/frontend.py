from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nnvm
import nnvm.graph
from nnvm.compiler import graph_attr, graph_util
import re

FRAME_SUPPORTED = ('tensorflow', 'keras', 'mxnet', 'onnx')
ATTR_NEED_CONVERT = ('channels', 'dilation', 'kernel_size',
                     'padding', 'strides', 'pool_size',
                     'axis', 'pad_value', 'pad_width',
                     '__shape__', 'shape', 'groups')


def _str_to_list(src):
    matchs = re.findall(r'([\[|\(]-*?\d+.*?[\]|\)])', src)
    ret = []
    for match in matchs:
        sub_matchs = re.findall(r'(-*?\d+)', match)
        ret.append([i for i in map(int, sub_matchs)])
    if len(ret) == 0:
        ret = int(src)
    elif len(ret) == 1:
        ret = ret[0]
    return ret

def _attrstr_to_number(graph):
    index = graph.index
    for node in index.nodes:
        if 'attrs' not in node:
            continue
        for attr_name, attr_val in node['attrs'].items():
            if attr_name in ATTR_NEED_CONVERT:
                node['attrs'][attr_name] = _str_to_list(attr_val)


def to_nnvm(graph, shape_dict, layout, mode='tensorflow'):
    """convert frontend graph to nnvm graph"""
    assert mode in FRAME_SUPPORTED
    if mode == 'tensorflow':
        sym, params = nnvm.frontend.from_tensorflow(graph, layout=layout, shape=shape_dict)
    elif mode == 'keras':
        sym, params = nnvm.frontend.from_keras(graph)
    elif mode == 'mxnet':
        sym, params = nnvm.frontend.from_mxnet(graph)
        sym = nnvm.sym.softmax(sym)
    else:
        sym, params = nnvm.frontend.from_onnx(graph)
    nnvm_graph = nnvm.graph.create(sym)

    _attrstr_to_number(nnvm_graph)

    return nnvm_graph, params
