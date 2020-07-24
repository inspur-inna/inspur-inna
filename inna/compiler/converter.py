from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _parse_inputs(graph):
    for node in graph.index.nodes:
        inputs = node['inputs']
        if len(inputs) > 0:
            node['inputs'] = [inp[0] for inp in node['inputs']]


def _remove_null_op_inputs(graph, shape_dict):
    index = graph.index
    for node in index.nodes:
        if node['name'] in shape_dict:
            node['op'] = 'input'
        if node['op'] != 'null':
            node['inputs'] = [inp for inp in node['inputs'] if index.nodes[inp]['op'] != 'null']


def _get_pad_width(pad_width, layout):
    return [pad_width[2][0], pad_width[3][0]] \
        if layout == 'NCHW' else [pad_width[1][0], pad_width[2][0]]

def _fuse_ops(graph, shape_dict, layout):
    hardware_graph = []
    new_index = 0
    index_map = {}
    index = graph.index
    # deal with batch_norm、pad
    for node in index.nodes:
        if node['op'] == 'conv2d':
            node['attrs']['batch_norm'] = False
        if node['op'] == 'batch_norm':
            index.nodes[node['inputs'][0]]['attrs']['batch_norm'] = True
        input_nodes = [index.nodes[inp] for inp in node['inputs']]
        for i, inp_node in enumerate(input_nodes):
            if inp_node['op'] in ['batch_norm', 'pad']:
                node['inputs'][i] = inp_node['inputs'][0]
            if inp_node['op'] == '__mul_scalar__':
                if len(inp_node['inputs']) == 0:
                    del node['inputs'][i]
                else:
                    node['inputs'][i] = inp_node['inputs'][0]
            if inp_node['op'] == 'pad':
                padding = _get_pad_width(inp_node['attrs']['pad_width'], layout)
                node['attrs']['padding'] = padding

    # remove null、batch_norm、pad op
    for i, node in enumerate(index.nodes):
        if node['op'] not in ['null', 'batch_norm', 'pad', '__mul_scalar__']:
            hardware_graph.append(node)
            index_map[i] = new_index
            new_index += 1
    # reset node inputs
    for node in hardware_graph:
        inputs = [index_map[inp] for inp in node['inputs']] if node['op'] != 'input' else []
        node['inputs'] = inputs
    return hardware_graph


def _layout_to_nchw(shape, layout):
    if len(shape) != 4:
        return shape
    return [shape[0], shape[3], shape[1], shape[2]] if layout == 'NHWC' else shape

def _parse_shape_to_hchw(hwgraph, shape_dict, layout):
    for node in hwgraph:
        node['shape'] = []
        node_name = node['name']
        if node_name in shape_dict:
            node['shape'] = _layout_to_nchw(shape_dict[node_name], layout)
            continue

        if 'attrs' not in node:
            continue
        attrs = node['attrs']
        if 'shape' in node['attrs']:
            node['shape'] = _layout_to_nchw(attrs['shape'], layout)
        elif '__shape__' in node['attrs']:
            node['shape'] = _layout_to_nchw(attrs['__shape__'], layout)


def _clac_shape(in_shape, kernel, pad, stride, channels):
    oheight = int((in_shape[2] + 2*pad[0] - kernel[0]) / stride[0] + 1)
    owidth = int((in_shape[3] + 2*pad[1] - kernel[1]) / stride[1] + 1)
    return [in_shape[0], channels, oheight, owidth]

def _infer_shape(hwgraph):
    for i, node in enumerate(hwgraph):
        if node['op'] == 'input':
            continue

        first_input = node['inputs'][0]
        input_shape = hwgraph[first_input]['shape']
        node['input_shape'] = input_shape

        if node['op'] == 'conv2d':
            attrs = node['attrs']
            assert len(input_shape) > 0
            node['shape'] = _clac_shape(input_shape, attrs['kernel_size'],
                                        attrs.get('padding', [0, 0]), attrs['strides'], attrs['channels'])
        elif node['op'] in ['max_pool2d', 'avg_pool2d']:
            attrs = node['attrs']
            padding = attrs.get('padding', [0, 0])
            node['shape'] = _clac_shape(input_shape, attrs['pool_size'],
                                        padding, attrs['strides'], input_shape[1])
        elif node['op'] in ['global_avg_pool2d', 'mean']:
            node['shape'] = [input_shape[0], input_shape[1], 1, 1]
        elif node['op'] == 'flatten':
            node['shape'] = input_shape[:2]
        elif node['op'] == 'dense':
            # TODO
            if len(input_shape) == 0:
                node['shape'] = [1, node['attrs']['units']]
                continue
            node['shape'] = [input_shape[0], node['attrs']['units']]
        elif node['op'] in ['pad', 'batch_norm', 'relu', 'elemwise_add', 'broadcast_add', 'softmax']:
            node['shape'] = input_shape

    # remove input node
    new_hwgraph = []
    index_map = {}
    new_index = 0
    for i, node in enumerate(hwgraph):
        if node['op'] != 'input':
            new_hwgraph.append(node)
            index_map[i] = new_index
            new_index += 1
    # reset node inputs
    for i, node in enumerate(new_hwgraph):
        if hwgraph[node['inputs'][0]]['op'] == 'input':
            inputs = []
        else:
            inputs = [index_map[inp] for inp in node['inputs']]
        node['inputs'] = inputs
    return new_hwgraph


def convert(graph, shape_dict, layout):
    """convert nnvm graph to hardware supported graph"""
    _parse_inputs(graph)

    _remove_null_op_inputs(graph, shape_dict)

    hwgraph = _fuse_ops(graph, shape_dict, layout)

    _parse_shape_to_hchw(hwgraph, shape_dict, layout)

    hwgraph = _infer_shape(hwgraph)

    return hwgraph
