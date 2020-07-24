from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tvm
import tvm.relay

import re
import json

FRAME_SUPPORTED = ('tensorflow', 'keras', 'mxnet', 'onnx')

ATTR_NEED_CONVERT = ("num_outputs", "num_inputs", "flatten_data")


def _attrstr_to_number(graph):
    nodes = graph['nodes']
    for node in nodes:
        if 'attrs' not in node:
            continue
        for attr_name, attr_val in node['attrs'].items():
            if attr_name in ATTR_NEED_CONVERT:
                node['attrs'][attr_name] = int(attr_val)

def to_tvm(graph, shape_dict, layout, mode='tensorflow'):
    """convert frontend graph to nnvm graph"""
    assert mode in FRAME_SUPPORTED
    if mode == 'tensorflow':
        mod, params = tvm.relay.frontend.from_tensorflow(graph, layout=layout, shape=shape_dict)
    elif mode == 'keras':
        mod, params = tvm.relay.frontend.from_keras(graph)
    elif mode == 'mxnet':
        mod, params = tvm.relay.frontend.from_mxnet(graph)
    else:
        mod, params = tvm.relay.frontend.from_onnx(graph)

    mod = tvm.relay.transform.InferType()(mod)

    target = 'llvm'
    target_host = 'llvm'
    with tvm.relay.build_config(opt_level=0):
        tvm_graph_json, lib, params = tvm.relay.build(mod, target=target, target_host=target_host, params=params)

    #with open("./json/resnet_v1_50_tvm_0.json", 'w') as fp:
        #fp.write(tvm_graph)
    tvm_graph = json.loads(tvm_graph_json)
    _attrstr_to_number(tvm_graph)

    return tvm_graph, params
