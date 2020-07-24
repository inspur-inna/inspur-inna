# uncompyle6 version 3.3.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.7.1 (default, Dec 14 2018, 19:28:38) 
# [GCC 7.3.0]
# Embedded file name: ./xfdnn/tools/quantize/quantize_base.py
# Compiled at: 2018-10-13 08:25:54
import numpy as np, collections, json
#import matplotlib.pyplot as plt

def Float2Fixed2Float(data, bitwidth, threshold, f):
    if np.isclose(threshold, 0.0):
        threshold = np.zeros_like(threshold)
    scaling_factor = threshold / (pow(2, bitwidth - 1) - 1)
    orig = np.array(data)
    data = np.clip(data, -threshold, threshold)
    if threshold != 0:
        data /= scaling_factor
        data = f(data)
        data *= scaling_factor
    size = data.size
    #print('average error(orig-data):' ,error/size)
    return data

def CdfMeasure(x, y, measure_name):
    if False:
        pass
    else:
        if measure_name == 'Kullback-Leibler-J':
            return np.sum((x - y) * np.log2(x / y))
        return CdfMeasure(x, y, 'Kullback-Leibler-J')
    
def ComputeThreshold(data, bitwidth, bins):
    mn = 0
    mx = np.abs(data).max()
    print ('Min: ', np.min(data), ', Max: ', np.max(data))
    zed = np.float64(0.0)
    if np.isclose(mx, zed):
        th_layer_out = zed
        sf_layer_out = zed
        return th_layer_out
    #print('computeThreshold-bins:', bins)
    hist, bin_edges = np.histogram(np.abs(data), bins, range=(mn, mx), density=True)
    hist = hist / np.sum(hist)
    cumsum = np.cumsum(hist)

    n = pow(2, bitwidth - 1)
    threshold = []
    scaling_factor = []
    d = []
    if n + 1 > len(bin_edges) - 1:
        th_layer_out = bin_edges[-1]
        sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
        return th_layer_out
    for i in range(n + 1, len(bin_edges), 1):
        threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
        threshold = np.concatenate((threshold, [threshold_tmp]))
        scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
        scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
        p = np.copy(cumsum)
        p[(i - 1):] = 1
        x = np.linspace(0.0, 1.0, n)
        xp = np.linspace(0.0, 1.0, i)
        fp = p[:i]
        p_interp = np.interp(x, xp, fp)
        x = np.linspace(0.0, 1.0, i)
        xp = np.linspace(0.0, 1.0, n)
        fp = p_interp
        q_interp = np.interp(x, xp, fp)
        q = np.copy(p)
        q[:i] = q_interp
        d_tmp = CdfMeasure(cumsum, q, 'Kullback-Leibler-J')
        d = np.concatenate((d, [d_tmp]))

    th_layer_out = threshold[np.argmin(d)]
    sf_layer_out = scaling_factor[np.argmin(d)]

    assert type(th_layer_out) == np.float64
    return th_layer_out


def ThresholdLayerInputs(data, bitwidth):
    return np.max(np.abs(data))


def ThresholdWeights(data, bitwidth):
    threshold = np.max(np.abs(data), axis=tuple(range(1, data.ndim)))
    return threshold

def ThresholdWeights_per_layer(data, bitwidth):
    threshold = np.max(np.abs(data))
    print ('weight max threshold: ', threshold)
    return threshold

def ThresholdBiases(data, bitwidth):
    threshold = np.max(np.abs(data), axis=tuple(range(1, data.ndim)))
    return threshold


def ThresholdLayerOutputs(data, bitwidth):
    return ComputeThreshold(data.flatten(), bitwidth, 'sqrt')
    #return ComputeThreshold(data.flatten(), bitwidth, 2048) #this is ok!


def QuantizeBlob(data, bitwidth):
    threshold = ThresholdLayerOutputs(data, bitwidth)
    return (
     Float2Fixed2Float(data, bitwidth, threshold, np.round), threshold)

def QuantizeBlob_Fixed_threshold(data,bitwidth,threshold):
    return Float2Fixed2Float(data, bitwidth,threshold,np.round)


def QuantizeThresholdBlob(data, bitwidth, threshold):
    assert type(threshold) in [np.float32, np.float64], 'Theshold is not a scalar'
    return Float2Fixed2Float(data, bitwidth, threshold, np.round)


def QuantizeWeights_per_layer(threshold, bitwidth, data, mode='caffe'):
    if mode == 'tf':
        data = data.transpose(2, 3, 1, 0)
    for i in range(len(data)):
        data[i] = Float2Fixed2Float(data[i], bitwidth, threshold, np.round)
        
    if mode == 'tf':
        data = data.transpose(3, 2, 0, 1)
    return data

def QuantizeWeights(threshold, bitwidth, data, mode='caffe'):
    if mode == 'tf':
        data = data.transpose(2, 3, 1, 0)
    assert data.shape[0] == threshold.shape[0], 'Threshold shape does not match weight data shape'
    for i in range(len(threshold)):
        data[i] = Float2Fixed2Float(data[i], bitwidth, threshold[i], np.round)

    if mode == 'tf':
        data = data.transpose(3, 2, 0, 1)
    return data


class QuantParam:
    bw_layer_in = None
    th_layer_in = None
    bw_layer_out = None
    th_layer_out = None
    bw_params = None
    th_params = None

    def __init__(self):
        self.bw_layer_in = collections.OrderedDict()
        self.th_layer_in = collections.OrderedDict()
        self.bw_layer_out = collections.OrderedDict()
        self.th_layer_out = collections.OrderedDict()
        self.bw_params = collections.OrderedDict()
        self.th_params = collections.OrderedDict()

    def jsontype(self,quant_layers):
        json_payload = {}
        bw_layer_in = collections.OrderedDict()
        bw_layer_out = collections.OrderedDict()
        bw_params = collections.OrderedDict()
        sf_layer_in = collections.OrderedDict()
        sf_layer_out = collections.OrderedDict()
        sf_params = collections.OrderedDict()

        for name, layer_type in quant_layers:
            bw_layer_in[name] = self.bw_layer_in[name]
            bw_layer_out[name] = self.bw_layer_out[name]
            bw_params[name] = self.bw_params[name]
            sf_layer_in[name] = self.th_layer_in[name] / (pow(2, self.bw_layer_in[name] - 1) - 1)
            sf_layer_out[name] = self.th_layer_out[name] / (pow(2, self.bw_layer_out[name] - 1) - 1)
            sf_params[name] = self.th_params[name] / (pow(2, self.bw_params[name] - 1) - 1)

            # modify paprams based on 2
            sf_layer_in[name] = np.power(2, np.ceil(np.log2(sf_layer_in[name])))
            self.th_layer_in[name] = sf_layer_in[name] * (pow(2, self.bw_layer_in[name] - 1) - 1)

            sf_layer_out[name] = np.power(2, np.ceil(np.log2(sf_layer_out[name])))
            self.th_layer_out[name] = sf_layer_out[name] * (pow(2, self.bw_layer_in[name] - 1) - 1)

            sf_params[name] = np.power(2, np.ceil(np.log2(sf_params[name])))  # round
            self.th_params[name] = sf_params[name] * (pow(2, self.bw_layer_in[name] - 1) - 1)

            multiplier = sf_layer_in[name] * sf_params[name] / sf_layer_out[name]  # round
            canonical_factor = np.power(2, np.ceil(np.log2(multiplier)))

            json_payload[name] = {
                'bw_layer_in': self.bw_layer_in[name],
                'bw_layer_out': self.bw_layer_out[name],
                'bw_params': self.bw_params[name],
                'th_layer_in': self.th_layer_in[name].tolist(),
                'th_layer_out': self.th_layer_out[name].tolist(),
                'th_params': self.th_params[name].tolist(),
                'sf_layer_in': sf_layer_in[name].tolist(),
                'sf_layer_out': sf_layer_out[name].tolist(),
                'sf_params': sf_params[name].tolist(),
                'multiplier': canonical_factor.tolist()}
        return json_payload

    def saveToJson_per_layer(self, quant_layers, fname):
        json_payload = {}
        json_payload['network'] = []
        bw_layer_in = collections.OrderedDict()
        bw_layer_out = collections.OrderedDict()
        bw_params = collections.OrderedDict()
        sf_layer_in = collections.OrderedDict()
        sf_layer_out = collections.OrderedDict()
        sf_params = collections.OrderedDict()

        print 'Writing output files to %s...' % fname
  
        with open(fname, 'w') as (g):
            for name, layer_type in quant_layers:
                bw_layer_in[name] = self.bw_layer_in[name]
                bw_layer_out[name] = self.bw_layer_out[name]
                bw_params[name] = self.bw_params[name]
                sf_layer_in[name] = self.th_layer_in[name] / (pow(2, self.bw_layer_in[name] - 1) - 1)
                sf_layer_out[name] = self.th_layer_out[name] / (pow(2, self.bw_layer_out[name] - 1) - 1)
                sf_params[name] = self.th_params[name] / (pow(2, self.bw_params[name] - 1) - 1)

                
		#modify paprams based on 2
                sf_layer_in[name] = np.power(2, np.ceil(np.log2(sf_layer_in[name])))
                self.th_layer_in[name] = sf_layer_in[name]*(pow(2, self.bw_layer_in[name] - 1) - 1)

                sf_layer_out[name] = np.power(2, np.ceil(np.log2(sf_layer_out[name])))
                self.th_layer_out[name] = sf_layer_out[name]*(pow(2, self.bw_layer_in[name] - 1) - 1)		

                sf_params[name] = np.power(2, np.ceil(np.log2(sf_params[name]))) #round
                self.th_params[name] = sf_params[name]*(pow(2, self.bw_layer_in[name] - 1) - 1)

                multiplier = sf_layer_in[name] * sf_params[name] / sf_layer_out[name]  #round
                canonical_factor = np.power(2, np.ceil(np.log2(multiplier)))
                
                json_payload['network'].append({'name': name})
                json_payload['network'].append({#'name': name, 
                   'bw_layer_in': self.bw_layer_in[name], 
                   'bw_layer_out': self.bw_layer_out[name], 
                   'bw_params': self.bw_params[name], 
                   'th_layer_in': self.th_layer_in[name].tolist(), 
                   'th_layer_out': self.th_layer_out[name].tolist(), 
                   'th_params': self.th_params[name].tolist(), 
                   'sf_layer_in': sf_layer_in[name].tolist(), 
                   'sf_layer_out': sf_layer_out[name].tolist(), 
                   'sf_params': sf_params[name].tolist(), 
                   'multiplier': canonical_factor.tolist()})

            json.dump(json_payload, g, indent=4, sort_keys=True)

