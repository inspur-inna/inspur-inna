# uncompyle6 version 3.3.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.7.1 (default, Dec 14 2018, 19:28:38) 
# [GCC 7.3.0]
# Embedded file name: ./xfdnn/tools/quantize/quantize_caffe.py
# Compiled at: 2018-10-13 08:25:54
import numpy as np, google.protobuf.text_format as tfmt, caffe
from quantize_base import *
#import matplotlib.pyplot as plt

class Caffe:
    _net = None
    _net_parameter = None
    _deploy_model = None
    _weights = None
    _dims = None

    def __init__(self, deploy_model, weights):
        self._deploy_model = deploy_model
        self._weights = weights
        self._net, self._net_parameter = self.declare_network(deploy_model, weights)
        blob_dict = self._net.blobs
        first_key = blob_dict.keys()[0]
        first_blob = blob_dict[first_key]
        self._dims = first_blob.data.shape
        self._input = first_key
        useGpu = False
        if useGpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()


    def getNetwork(self):
        return (
         self._net, self._net_parameter)

    def declareTransformer(self, transpose, channel_swap, raw_scale, mean_value, input_scale, dims):
        self._transformer = caffe.io.Transformer({self._input: self._net.blobs[self._input].data.shape})
        self._transformer.set_transpose(self._input, transpose)
        self._transformer.set_channel_swap(self._input, channel_swap)
        self._transformer.set_raw_scale(self._input, raw_scale)
        self._transformer.set_mean(self._input, mean_value)
        self._transformer.set_input_scale(self._input, input_scale)

    def initializeBatch(self, calibration_size, calibration_filenames, calibration_indices):
        dims = list(self._dims[-3:])
        self._net.blobs[self._input].reshape(calibration_size, *dims)
        for i in range(calibration_size):
            print ('Adding', calibration_filenames[calibration_indices[i]], 'to calibration batch.')
            data = caffe.io.load_image(calibration_filenames[calibration_indices[i]])
            self._net.blobs[self._input].data[i] = self._transformer.preprocess(self._input, data)

    def declare_network(self, deploy_model, weights):
        net = caffe.Net(deploy_model, weights, caffe.TEST)
        net_parameter = caffe.proto.caffe_pb2.NetParameter()
        return (
         net, net_parameter)

    def executeCalibration(self, bitwidths, deploy_model, quant_params):
        net, net_parameter = self.getNetwork()
        bw_layer_in_global = bitwidths[0]
        bw_params_global = bitwidths[1]
        bw_layer_out_global = bitwidths[2]
        quant_layers = []
        with open(deploy_model, 'r') as (f):
            tfmt.Merge(f.read(), net_parameter)
            for name, layer in net.layer_dict.items():
                print '-' * 80
                print 'Processing layer %d of %d' % (list(net._layer_names).index(name), len(list(net._layer_names)))
                print 'Layer Name:' + name + ' Type:' + layer.type
                print ('Inputs:', str(net.bottom_names[name]) + ', Outputs:', str(net.top_names[name]))
                if layer.type in ('Input', 'Data'):
                    print ('input/data quantization!')
                    q = QuantizeInputCaffe(self, name, layer.type, bw_layer_out_global, quant_params)
                    q.preProcess()
                    q.execute()
                    q.postProcess()
                elif layer.type in ('Convolution', ):
                    print ('convolution quantization!')
                    q = QuantizeConvolutionCaffe(self, name, layer.type, bw_layer_out_global, bw_params_global, quant_params)
                    q.preProcess()
                    q.execute()
                    q.postProcess()
                    quant_layers.append((name, layer.type))
    
                elif layer.type in ('InnerProduct', ):
                    quantize_inner_product(name, net, quant_params, bw_params_global, bw_layer_out_global)
                    quant_layers.append((name, layer.type))
                elif layer.type in ('ReLU', ):
                    #quantize_relu and quantize_relu_KL are the same functions.
                    quantize_relu_withoutKL(name, net, quant_params, bw_layer_out_global)
    
                elif layer.type in ('Pooling', ):
                    quantize_pooling(name, net, quant_params)
                elif layer.type in ('LRN', ):
                    quantize_lrn(name, net, quant_params)
                elif layer.type in ('Dropout', ):
                    quantize_dropout(name, net, quant_params)
                elif layer.type in ('Split', ):
                    quantize_split(name, net, quant_params)
                elif layer.type in ('Concat', ):
                    quantize_concat(name, net, quant_params, bw_layer_out_global)
                elif layer.type in ('Eltwise', ):     
                    quantize_eltwise_bottom(name, net, quant_params)
                elif layer.type in ('Softmax', 'SoftmaxWithLoss'):
                    print 'Passing'
                else:
                    print 'Error: Quantization of ' + layer.type + ' is not yet supported'

        return quant_layers


class LayerExecute:
    _nodename = None
    _layertype = None

    def __init__(self, nodename, layertype):
        self._nodename = nodename
        self._layertype = layertype

    def preProcess(self):
        pass

    def execute(self):
        pass

    def postProcess(self):
        pass


class LayerExecuteCaffe(LayerExecute):
    _topname = None
    _quant_params = None
    _net = None
    _bitwidth = None

    def __init__(self, caffeobj, nodename, layertype, bitwidth, quant_params):
        LayerExecute.__init__(self, nodename, layertype)
        self._net = caffeobj.getNetwork()[0]
        self._topname = self._net.top_names[nodename][0]
        self._bitwidth = bitwidth
        self._quant_params = quant_params


class QuantizeInputCaffe(LayerExecuteCaffe):

    def __init__(self, caffeobj, nodename, layertype, bitwidth, quant_params):
        LayerExecuteCaffe.__init__(self, caffeobj, nodename, layertype, bitwidth, quant_params)

    def preProcess(self):
        pass

    def execute(self):
        self._net.forward(start=self._nodename, end=self._nodename)

    def postProcess(self):
        print 'Quantizing layer output...'
        threshold = np.float64(254.00)

        self._quant_params.bw_layer_out[self._nodename] = self._bitwidth
        self._quant_params.th_layer_out[self._nodename] = threshold
        print (
         'bw_layer_out: ', self._quant_params.bw_layer_out[self._nodename])
        print ('th_layer_out: ', self._quant_params.th_layer_out[self._nodename])


class QuantizeConvolutionCaffe(LayerExecuteCaffe):
    _botname = None
    _bitwidth_params = None

    def __init__(self, caffeobj, nodename, layertype, bitwidth, bitwidth_params, quant_params):
        LayerExecuteCaffe.__init__(self, caffeobj, nodename, layertype, bitwidth, quant_params)
        self._botname = self._net.bottom_names[self._nodename][0]
        self._bitwidth_params = bitwidth_params

    def preProcess(self):
        print (
         'Quantizing conv input layer ...', self._topname)
        bitwidth = self._quant_params.bw_layer_out[self._botname]
        threshold = self._quant_params.th_layer_out[self._botname]
       
        self._quant_params.bw_layer_in[self._nodename] = bitwidth
        self._quant_params.th_layer_in[self._nodename] = threshold
        print 'Quantizing conv weights for layer %s...' % self._nodename
        bitwidth = self._bitwidth_params
        threshold = ThresholdWeights_per_layer(self._net.params[self._nodename][0].data, bitwidth)

        #threshold = ThresholdWeights_KL(self._net.params[self._nodename][0].data, bitwidth)  
        #per channel
        #threshold = ThresholdWeights(self._net.params[self._nodename][0].data, bitwidth)

        
        self._quant_params.bw_params[self._nodename] = bitwidth
        self._quant_params.th_params[self._nodename] = threshold
	
        self._net.params[self._nodename][0].data[:] = QuantizeWeights_per_layer(threshold, bitwidth, self._net.params[self._nodename][0].data[:])

        #self._net.params[self._nodename][0].data[:] = QuantizeWeights(threshold, bitwidth, self._net.params[self._nodename][0].data[:])

    def execute(self):
        self._net.forward(start=self._nodename, end=self._nodename)

    def postProcess(self):
        data = self._net.blobs[self._nodename].data[...]
        maxValue = np.abs(data).max()
        minValue = np.min(data)
        print('max of convolution:', maxValue)
        threshold = ThresholdLayerOutputs(data, self._bitwidth)
        
        #####plot histogram
	hist, bin_edges = np.histogram(data, 2048, range=(0, maxValue), density=True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        index = np.arange(len(bin_edges)-1)
        plt.figure()
        plt.bar(bin_edges[0:len(bin_edges)-1], cumsum)
        histName = self._nodename + '.png'
        plt.savefig(histName)
        #####
        
        self._quant_params.bw_layer_out[self._nodename] = self._bitwidth
        self._quant_params.th_layer_out[self._nodename] = threshold
	
        print (
         'bw_layer_in: ', self._quant_params.bw_layer_in[self._nodename])
        print ('th_layer_in: ', self._quant_params.th_layer_in[self._nodename])
        print (
         'bw_layer_out: ', self._quant_params.bw_layer_out[self._nodename])
        print ('th_layer_out: ', self._quant_params.th_layer_out[self._nodename])


def quantize_inner_product(name, net, quant_params, bw_params_global, bw_layer_out_global):
    topname = net.top_names[name][0]
    botname = net.bottom_names[name][0]
    data = net.blobs[botname].data[...]
    bitwidth = quant_params.bw_layer_out[botname]
    threshold = quant_params.th_layer_out[botname]
    
    quant_params.bw_layer_in[topname] = bitwidth
    quant_params.th_layer_in[topname] = threshold

    #delete 2019/7/15
    #net.blobs[botname].data[:] = Float2Fixed2Float(data, bitwidth, threshold, np.round)
    data = net.params[name][0].data[...]

    bitwidth = bw_params_global

    threshold = ThresholdWeights_per_layer(data, bitwidth) #lly per layer
    #threshold = ThresholdWeights(data, bitwidth) #per channels
    
    quant_params.bw_params[topname] = bitwidth
    quant_params.th_params[topname] = threshold

    #per-channel
    #for i in range(len(quant_params.th_params[topname])):
    #    net.params[name][0].data[i] = Float2Fixed2Float(data[i], bitwidth, threshold[i], np.round)

    net.forward(start=name, end=name)
    data = net.blobs[topname].data[...]
    bitwidth = bw_layer_out_global
    threshold = ThresholdLayerOutputs(data, bitwidth)
    
    quant_params.bw_layer_out[topname] = bitwidth
    quant_params.th_layer_out[topname] = threshold
    print (
     'bw_layer_in: ', quant_params.bw_layer_in[topname])
    print ('th_layer_in: ', quant_params.th_layer_in[topname])
    print (
     'bw_layer_out: ', quant_params.bw_layer_out[topname])
    print ('th_layer_out: ', quant_params.th_layer_out[topname])


def quantize_relu(name, net, quant_params, bw_layer_out_global):
    topname = net.top_names[name][0]
    botname = net.bottom_names[name][0]
    net.forward(start=botname, end=name)
    net.forward(start=name, end=name)
    data = net.blobs[topname].data[...]
    maxValue = np.max(abs(data))
    print('max of relu:', maxValue)
    bitwidth = bw_layer_out_global
    threshold = ThresholdLayerOutputs(data, bitwidth)  #KL
    quant_params.bw_layer_out[topname] = bitwidth
    quant_params.th_layer_out[topname] = threshold
    print (
     'bw_layer_out: ', quant_params.bw_layer_out[topname])
    print ('th_layer_out: ', quant_params.th_layer_out[topname])


def quantize_relu_withoutKL(name, net, quant_params, bw_layer_out_global):
    topname = net.top_names[name][0]
    print('topname:', topname)
    botname = net.bottom_names[name][0]
    print('botname:', botname)
    net.forward(start=botname, end=name)
    net.forward(start=name, end=name)
    quant_params.bw_layer_out[topname] = bw_layer_out_global
    quant_params.th_layer_out[topname] = quant_params.th_layer_out[botname]
    print('bw_layer_out:', quant_params.bw_layer_out[topname])
    print('th_layer_out:', quant_params.th_layer_out[topname])
    

def quantize_pooling(name, net, quant_params):
    topname = net.top_names[name][0]
    botname = net.bottom_names[name][0]
    data = net.blobs[botname].data[...]
    bitwidth = quant_params.bw_layer_out[botname]
    threshold = quant_params.th_layer_out[botname]
    quant_params.bw_layer_in[topname] = bitwidth
    quant_params.th_layer_in[topname] = threshold
    net.forward(start=name, end=name)
    data = net.blobs[topname].data[...]
    bitwidth = quant_params.bw_layer_in[topname]
    threshold = quant_params.th_layer_in[topname]
    quant_params.bw_layer_out[topname] = bitwidth
    quant_params.th_layer_out[topname] = threshold
    print (
     'bw_layer_in: ', quant_params.bw_layer_in[topname])
    print ('th_layer_in: ', quant_params.th_layer_in[topname])
    print (
     'bw_layer_out: ', quant_params.bw_layer_out[topname])
    print ('th_layer_out: ', quant_params.th_layer_out[topname])


def quantize_lrn(name, net, quant_params):
    topname = net.top_names[name][0]
    botname = net.bottom_names[name][0]
    data = net.blobs[botname].data[...]
    bitwidth = quant_params.bw_layer_out[botname]
    threshold = quant_params.th_layer_out[botname]
    quant_params.bw_layer_in[topname] = bitwidth
    quant_params.th_layer_in[topname] = threshold
    net.forward(start=name, end=name)
    data = net.blobs[topname].data[...]
    bitwidth = quant_params.bw_layer_in[topname]
    threshold = quant_params.th_layer_in[topname]
    quant_params.bw_layer_out[topname] = bitwidth
    quant_params.th_layer_out[topname] = threshold
    print (
     'bw_layer_in: ', quant_params.bw_layer_in[topname])
    print ('th_layer_in: ', quant_params.th_layer_in[topname])
    print (
     'bw_layer_out: ', quant_params.bw_layer_out[topname])
    print ('th_layer_out: ', quant_params.th_layer_out[topname])


def quantize_dropout(name, net, quant_params):
    topname = net.top_names[name][0]
    botname = net.bottom_names[name][0]
    data = net.blobs[botname].data[...]
    bitwidth = quant_params.bw_layer_out[botname]
    threshold = quant_params.th_layer_out[botname]
    quant_params.bw_layer_in[topname] = bitwidth
    quant_params.th_layer_in[topname] = threshold
    net.forward(start=name, end=name)
    data = net.blobs[topname].data[...]
    bitwidth = quant_params.bw_layer_in[topname]
    threshold = quant_params.th_layer_in[topname]
    quant_params.bw_layer_out[topname] = bitwidth
    quant_params.th_layer_out[topname] = threshold
    print (
     'bw_layer_in: ', quant_params.bw_layer_in[topname])
    print ('th_layer_in: ', quant_params.th_layer_in[topname])
    print (
     'bw_layer_out: ', quant_params.bw_layer_out[topname])
    print ('th_layer_out: ', quant_params.th_layer_out[topname])


def quantize_split(name, net, quant_params):
    topname = net.top_names[name][0]
    botname = net.bottom_names[name][0]
    for i in range(len(net.top_names[name])):
        topnamei = net.top_names[name][i]
        quant_params.bw_layer_in[topnamei] = quant_params.bw_layer_out[botname]
        quant_params.th_layer_in[topnamei] = quant_params.th_layer_out[botname]

    net.forward(start=name, end=name)
    for i in range(len(net.top_names[name])):
        topnamei = net.top_names[name][i]
        quant_params.bw_layer_out[topnamei] = quant_params.bw_layer_in[topnamei]
        quant_params.th_layer_out[topnamei] = quant_params.th_layer_in[topnamei]

    print ('bw_layer_in: ', quant_params.bw_layer_in[topname])
    print ('th_layer_in: ', quant_params.th_layer_in[topname])
    print (
     'bw_layer_out: ', quant_params.bw_layer_out[topname])
    print ('th_layer_out: ', quant_params.th_layer_out[topname])


def quantize_concat(name, net, quant_params, bw_layer_out_global):
    topname = net.top_names[name][0]
    botname = net.bottom_names[name][0]
    for bottom_name in net.bottom_names[name]:
        start_name = list((bottom_name in net.top_names[name] and name for name, layer in net.layer_dict.items()))[0]
        end_name = list((bottom_name in net.top_names[name] and name for name, layer in net.layer_dict.items()))[-1]
        net.forward(start=start_name, end=end_name)

    net.forward(start=name, end=name)
    data = net.blobs[topname].data[...]
    bitwidth = bw_layer_out_global
    threshold = ThresholdLayerOutputs(data, bitwidth)
    quant_params.bw_layer_out[topname] = bitwidth
    quant_params.th_layer_out[topname] = threshold
    #net.blobs[topname].data[:] = Float2Fixed2Float(data, bitwidth, threshold, np.round)
    quant_params.bw_layer_in[topname] = quant_params.bw_layer_out[topname]
    quant_params.th_layer_in[topname] = quant_params.th_layer_out[topname]
    for bottom_name in net.bottom_names[name]:
        quant_params.bw_layer_out[net.top_names[bottom_name][0]] = quant_params.bw_layer_in[topname]
        quant_params.th_layer_out[net.top_names[bottom_name][0]] = quant_params.th_layer_in[topname]

    print ('bw_layer_in: ', quant_params.bw_layer_in[topname])
    print ('th_layer_in: ', quant_params.th_layer_in[topname])
    print (
     'bw_layer_out: ', quant_params.bw_layer_out[topname])
    print ('th_layer_out: ', quant_params.th_layer_out[topname])


# this method is right.
def quantize_eltwise_bottom(name, net, quant_params):
    topname = net.top_names[name][0]
    botname = net.bottom_names[name][0]
    bitwidth = quant_params.bw_layer_out[botname]
    threshold = quant_params.th_layer_out[botname]
    for i in range(1, len(net.bottom_names[name])):
        bitwidth = np.maximum(bitwidth, quant_params.bw_layer_out[net.bottom_names[name][i]])
        threshold = np.maximum(threshold, quant_params.th_layer_out[net.bottom_names[name][i]])
    
    quant_params.bw_layer_in[topname] = bitwidth
    quant_params.th_layer_in[topname] = threshold

    for i in range(0, len(net.bottom_names[name])):
        quant_params.th_layer_out[net.bottom_names[name][i]] = threshold

    for i in range(0, len(net.bottom_names[name])):   
        net.blobs[net.bottom_names[name][i]].data[:] = QuantizeThresholdBlob(net.blobs[net.bottom_names[name][i]].data[:], bitwidth, threshold)

    net.forward(start=name, end=name)
    quant_params.bw_layer_out[topname] = bitwidth
    quant_params.th_layer_out[topname] = threshold

    print (
     'bw_layer_in: ', quant_params.bw_layer_in[topname])
    print ('th_layer_in: ', quant_params.th_layer_in[topname])
    print (
     'bw_layer_out: ', quant_params.bw_layer_out[topname])
    print ('th_layer_out: ', quant_params.th_layer_out[topname])


def quantize_softmax(name, net, quant_params):
    topname = net.top_names[name][0]
    botname = net.bottom_names[name][0]
    data = net.blobs[botname].data[...]
    bw_layer_in[topname] = bw_layer_out[botname]
    th_layer_in[topname] = th_layer_out[botname]
    net.forward(start=name, end=name)
# okay decompiling quantize_caffe.pyc
