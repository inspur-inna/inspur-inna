from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tvm.relay.testing.tf as tf_testing
import os


def resnet_v1_50():
    with tf.Graph().as_default():
        model = {
            'pb': '../models/tensorflow/resnet/resnet50_without_bn_test.pb',
            'shape_dict': {
                'input': (1, 224, 224, 3), 
            },
            'layout': 'NHWC',
            'out_node': 'prob', 
        }
        with tf.gfile.GFile(model['pb'], 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')
            # Call the utility to import the graph definition into default graptributeError: module 'tensorflow.tools.api.generator.api.compat' has no attribute 'v1'h.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)

            with tf.Session() as sess:
                # Add shapes to the graph.
                graph_def = tf_testing.AddShapesToGraphDef(sess, model['out_node'])

    return graph_def, model['shape_dict'], model['layout']
