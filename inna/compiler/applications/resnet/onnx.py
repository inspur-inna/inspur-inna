from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import onnx


def resnet_v1_50():
    graph = onnx.load('../models/onnx/resnet/restnet50_v1.1.onnx')
    shape_dict = {
        'gpu_0/data_0': (1, 3, 224, 224),
    }
    layout = 'NCHW'

    return graph, shape_dict, layout
