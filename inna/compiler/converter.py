from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _parse_inputs(graph):
    shape_list = graph['attrs']['shape'][1]
    for i, node in enumerate(graph['nodes']):
        #print(node)
        inputs = node['inputs']
        node['shape'] = []
        if len(inputs) > 0:
            node['inputs'] = [inp[0] for inp in node['inputs']]
            node['shape'] = [shape_list[j] for j in node['inputs']]
        node['shape'].append(shape_list[i])


def _remove_null_op_inputs(graph, shape_dict):
    for node in graph['nodes']:
        if node['name'] in shape_dict:
            node['op'] = 'input'
        if node['op'] != 'null':
            node['inputs'] = [inp for inp in node['inputs'] if graph['nodes'][inp]['op'] != 'null']


def _parse_op(graph):
    for node in graph['nodes']:
        if node['name'] in ['data', 'input']:
            node['op'] = node['name']
            continue
        if node['op'] in ['null']:
            continue
        #print(node)
        name_str = node['attrs']['func_name']
        name_list = name_str.split('_')
        node['op'] = ''
        for op_name in name_list:
            if (op_name not in ['fused','nn']) and (op_name.isdigit()==False):
                node['op'] = node['op'] + op_name + '_'
        node['op'] = node['op'].strip('_')
        if node['op'] in ['add']:
            if len(node['inputs']) > 1:
                i = 0
                for j in node['inputs']:
                    if graph['nodes'][j]['op'] == 'null':
                        break
                    i += 1
                if i == len(node['inputs']):
                    node['op'] = 'eltwise_add'
        if node['op'] in ['transpose']:
            node['op'] = 'null'
        #if node['op'] in ['pad1','pad2','pad3','pad4','pad5']:
            #node['op'] = 'pad'
        #if node['op'] in ['relu1','relu2']:
            #node['op'] = 'relu'


def _get_pad_width(pad_width, layout):
    return [pad_width[2][0], pad_width[3][0]] \
        if layout == 'NCHW' else [pad_width[1][0], pad_width[2][0]]

def _fuse_ops(graph, shape_dict, layout):
    hardware_graph = []
    new_index = 0
    index_map = {}
    # deal with batch_norm、pad
    for node in graph['nodes']:
        if node['op'] == 'conv2d':
            node['attrs']['batch_norm'] = False
            node['attrs']['use_bias'] = False
        if node['op'] == 'batch_norm':
            graph['nodes'][node['inputs'][0]]['attrs']['batch_norm'] = True
        if node['op'] == 'add':
            graph['nodes'][node['inputs'][0]]['attrs']['use_bias'] = True
        input_nodes = [graph['nodes'][inp] for inp in node['inputs']]
        for i, inp_node in enumerate(input_nodes):
            #print(inp_node)
            if inp_node['op'] in ['batch_norm', 'pad', 'add']:
                node['inputs'][i] = inp_node['inputs'][0]
            if inp_node['op'] == '__mul_scalar__':
                if len(inp_node['inputs']) == 0:
                    del node['inputs'][i]
                else:
                    node['inputs'][i] = inp_node['inputs'][0]
            #if inp_node['op'] == 'pad':
                #padding = _get_pad_width(inp_node['attrs']['pad_width'], layout)
                #node['attrs']['padding'] = padding

    # remove null、batch_norm、pad op
    for i, node in enumerate(graph['nodes']):
        if node['op'] not in ['null', 'batch_norm', 'pad', '__mul_scalar__', 'add', 'transpose']:
            hardware_graph.append(node)
            index_map[i] = new_index
            new_index += 1
    # reset node inputs
    for node in hardware_graph:
        inputs = [index_map[inp] for inp in node['inputs']] if node['op'] not in ['input','data'] else []
        node['inputs'] = inputs
    return hardware_graph


def _layout_to_nchw(shape, layout):
    if len(shape) != 4:
        return shape
    return [shape[0], shape[3], shape[1], shape[2]] if layout == 'NHWC' else shape

def _parse_shape_to_nchw(hwgraph, shape_dict, layout):
    '''
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
    '''
    for node in hwgraph:
        if 'input_shape' in node:
            node['input_shape'] = _layout_to_nchw(node['input_shape'], layout)
        if 'output_shape' in node:
            node['output_shape'] = _layout_to_nchw(node['output_shape'], layout)


def _clac_shape(in_shape, kernel, pad, stride, channels):
    oheight = int((in_shape[2] + 2*pad[0] - kernel[0]) / stride[0] + 1)
    owidth = int((in_shape[3] + 2*pad[1] - kernel[1]) / stride[1] + 1)
    return [in_shape[0], channels, oheight, owidth]

def _infer_shape(hwgraph):
    for i, node in enumerate(hwgraph):
        if node['op'] in ['input', 'data']:
            continue
        if len(node['inputs']) < 1:
            continue
        first_input = node['inputs'][0]
        input_shape = hwgraph[first_input]['shape'][-1]
        node['input_shape'] = input_shape
        output_shape = node['shape'][-1]
        node['output_shape'] = output_shape

        if node['op'] == 'conv2d':
            attrs = node['attrs']
            assert len(input_shape) > 0
            kernel_shape = node['shape'][1]
            attrs['kernel_size'] = [kernel_shape[0],kernel_shape[1]]
            attrs['channels'] = kernel_shape[3]
            attrs['padding'] = [int(kernel_shape[0]/2), int(kernel_shape[1]/2)]
            output_shape = node['shape'][-1]
            attrs['strides'] = [int(input_shape[1]/output_shape[1]), int(input_shape[2]/output_shape[2])]

            #node['shape'] = _clac_shape(input_shape, attrs['kernel_size'],
                                        #attrs.get('padding', [0, 0]), attrs['strides'], attrs['channels'])

        elif node['op'] in ['max_pool2d', 'avg_pool2d']:
            attrs = node['attrs']
            #padding = attrs.get('padding', [0, 0])

            if node['op'] in ['max_pool2d']:
                attrs['pool_size'] = [3, 3]
            else:
                attrs['pool_size'] = [7, 7]
            attrs['padding'] = [0, 0, int(attrs['pool_size'][0]/2), int(attrs['pool_size'][1]/2)]

            attrs['strides'] = [int(output_shape[1]/input_shape[1]), int(output_shape[2]/input_shape[2])]

            #node['shape'] = _clac_shape(input_shape, attrs['pool_size'],
                                        #attrs['padding'], attrs['strides'], input_shape[1])

        elif node['op'] in ['global_avg_pool2d', 'mean']:
            #node['output_shape'] = [input_shape[0], input_shape[1], 1, 1]
            continue
        elif node['op'] == 'flatten':
            #node['output_shape'] = input_shape[:2]
            continue
        elif node['op'] == 'dense':
            # TODO
            if len(input_shape) == 0:
                #node['output_shape'] = [1, node['attrs']['units']]
                continue
            #node['shape'] = [input_shape[0], node['attrs']['units']]

        elif node['op'] in ['pad', 'batch_norm', 'relu', 'eletwise_add', 'broadcast_add', 'softmax']:
            #node['shape'] = input_shape
            continue

    # remove input node
    new_hwgraph = []
    index_map = {}
    new_index = 0
    for i, node in enumerate(hwgraph):
        if node['op'] not in ['input', 'data']:
            new_hwgraph.append(node)
            index_map[i] = new_index
            new_index += 1
    # reset node inputs
    for i, node in enumerate(new_hwgraph):
        if len(node['inputs']) < 1:
            continue
        if hwgraph[node['inputs'][0]]['op'] in ['input', 'data']:
            inputs = []
        else:
            inputs = [index_map[inp] for inp in node['inputs']]
        node['inputs'] = inputs
    return new_hwgraph


def convert(graph, shape_dict, layout):
    """convert nnvm graph to hardware supported graph"""
    _parse_inputs(graph)

    _parse_op(graph)

    _remove_null_op_inputs(graph, shape_dict)

    hwgraph = _fuse_ops(graph, shape_dict, layout)

    hwgraph = _infer_shape(hwgraph)

    _parse_shape_to_nchw(hwgraph, shape_dict, layout)

    #print(hwgraph)

    return hwgraph