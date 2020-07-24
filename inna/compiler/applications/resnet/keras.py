from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras


def resnet_v1_50():
    graph = keras.applications.resnet50.ResNet50(include_top=True, weights=None,
                                                          input_shape=(224, 224, 3), classes=1000)
    graph.load_weights('../models/keras/resnet/resnet50_weights.h5')
    shape_dict = {
        'input_1': (1, 3, 224, 224),
    }
    layout = 'NCHW'

    return graph, shape_dict, layout
