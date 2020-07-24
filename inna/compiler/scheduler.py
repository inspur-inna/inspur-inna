from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# real max batch = MAX_HW_BATCH * HW_BATCH_SIZE
MAX_HW_BATCH = 8
HW_BATCH_SIZE = 4
DDR_ADDR_MASK = 8 * 1024**3 - 1
FILTER_START_ADDR = 7 * 1024**3
QUANT_START_ADDR = 6 * 1024**3
FEATURE_START_ADDR = [1 * 1024**3, 3 * 1024**3]


def _decide_exection_order(graph):
    for i, node in enumerate(graph):
        node['runid'] = i
        node['parents'] = node['inputs']
        del node['inputs']
        node['children'] = []
        node['siblings'] = []
    for node in graph:
        for parent in node['parents']:
            graph[parent]['children'].append(node['runid'])
    for node in graph:
        if len(node['children']) >= 2:
            children_node = [graph[child] for child in node['children']]
            for child_node in children_node:
                for child_id in node['children']:
                    if child_node['runid'] != child_id:
                        child_node['siblings'].append(child_id)


def _aligned(num, aliged):
    return int((num + aliged - 1) / aliged) + 1
def _shape_to_size(shape, maxbatch, hwbatch):
    # TODO
    if len(shape) != 4:
        return 0
    return int(_aligned(shape[1], 64) * shape[2] * shape[3] * maxbatch * hwbatch)

def _calc_inout_size(graph):
    for node in graph:
        print(node)
        node['input_size'] = _shape_to_size(node['input_shape'],
                                            MAX_HW_BATCH, HW_BATCH_SIZE)
        node['output_size'] = _shape_to_size(node['output_shape'], MAX_HW_BATCH, HW_BATCH_SIZE)


def _alloc_param_memory(graph):
    filter_addr = FILTER_START_ADDR
    quant_addr = QUANT_START_ADDR
    for node in graph:
        if node['op'] == 'conv2d':
            node['filter_addr'] = filter_addr
            attrs = node['attrs']
            filter_addr += _aligned(attrs['channels'], 64) * _aligned(node['output_shape'][1], 64) \
                           * attrs['kernel_size'][0] * attrs['kernel_size'][1]
        if node['op'] in ['conv2d', 'relu', 'max_pool2d', 'eltwise_add',
                          'boardcast_add', 'mean', 'global_avg_pool2d']:
            node['quant_addr'] = quant_addr
            quant_addr += 256 * 2

def _alloc_feature_memory(graph):
    feature_addr = [FEATURE_START_ADDR[0], FEATURE_START_ADDR[1]]
    for node in graph:
        node['input_addr'] = feature_addr[0] \
            if len(node['parents']) == 0 else graph[node['parents'][0]]['output_addr']
        iotype = node['runid'] % 2
        feature_addr[iotype] += node['input_size']
        node['output_addr'] = feature_addr[1 - iotype]
        feature_addr[1 - iotype] += node['output_size']
        if node['op'] == 'eltwise_add':
            node['residual_addr'] = graph[node['parents'][1]]['output_addr']

def _alloc_memory(graph):
    for node in graph:
        node['input_addr'] = DDR_ADDR_MASK
        node['output_addr'] = DDR_ADDR_MASK
        node['filter_addr'] = DDR_ADDR_MASK
        node['quant_addr'] = DDR_ADDR_MASK
        node['residual_addr'] = DDR_ADDR_MASK
    _alloc_param_memory(graph)
    _alloc_feature_memory(graph)


def _convert_to_assem_graph(graph):
    for node in graph:
        # TODO
        if node['op'] in ['squeeze', 'reshape', 'flatten', 'softmax', 'dense', 'transpose']:
            continue
        node['batch_size'] = HW_BATCH_SIZE
        _, node['input_channel'], node['input_height'], node['input_width'] = node['input_shape'] \
            if len(node['input_shape']) == 4 else node['input_shape'] + [1, 1]
        _, node['output_channel'], node['output_height'], node['output_width'] = node['output_shape'] \
            if len(node['output_shape']) == 4 else node['output_shape'] + [1, 1]
        if node['op'] == 'conv2d':
            node['instr_name'] = 'CONV'
            attrs = node['attrs']
            node['conv_type'] = (1 << 2) + (int(attrs['batch_norm']) << 1) + 0
            node['kernel_size'] = attrs['kernel_size'][0]
            padding = attrs.get('padding', [0, 0])
            node['h_pad'] = padding[0]
            node['v_pad'] = padding[1]
            node['stride'] = attrs['strides'][0]
        elif node['op'] == 'relu':
            node['instr_name'] = 'ACTIVE'
            node['active_type'] = 1
        elif node['op'] == 'max_pool2d':
            node['instr_name'] = 'POOL'
            node['pool_type'] = 0
            attrs = node['attrs']
            node['kernel_size'] = attrs['pool_size'][0]
            node['stride'] = attrs['strides'][0]
        elif node['op'] in ['eltwise_add', 'broadcast_add']:
            node['instr_name'] = 'ELTWISE'
            runid = min(node['parents'])
            if graph[runid]['parents'] >= graph[runid+1]['parents']:
                node['eltwise_type'] = 1
            else:
                node['eltwise_type'] = 2
        elif node['op'] in ['global_avg_pool2d', 'mean', 'avg_pool2d']:
            node['instr_name'] = 'POOL'
            node['pool_type'] = 1

def schedule(graph):
    """decide op execution order, alloc memory"""
    _decide_exection_order(graph)

    _calc_inout_size(graph)

    _alloc_memory(graph)

    _convert_to_assem_graph(graph)
