from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet.gluon.model_zoo.vision import get_model


def resnet_v1_50():
    graph = get_model('resnet50_v1', pretrained=True)
    shape_dict = {
        'data': (1, 3, 224, 224),
    }
    layout = 'NCHW'
    return graph, shape_dict, layout
