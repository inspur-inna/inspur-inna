# uncompyle6 version 3.3.5
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.7.0 (default, Jun 28 2018, 13:15:42) 
# [GCC 7.2.0]
# Embedded file name: ./xfdnn/tools/quantize/quantize_tf.py
# Compiled at: 2018-10-13 07:26:56
import argparse, numpy as np, tensorflow as tf, os.path as osp, threading, random, pdb, os
from quantize_base import *
from tensorflow.python.platform import gfile
messages = []


layer_name = {"fc1000/biases": 53, "fc1000/weights": 53,"res5c_branch2c/biases": 52,"res5c_branch2c/weights": 52, "res5c_branch2b/biases": 51,"res5c_branch2b/weights": 51,
              "res5c_branch2a/biases": 50, "res5c_branch2a/weights": 50, "res5b_branch2c/biases": 49, "res5b_branch2c/weights": 49, "res5b_branch2b/biases": 48, "res5b_branch2b/weights": 48,
              "res5b_branch2a/biases": 47, "res5b_branch2a/weights": 47, "res5a_branch2c/biases": 46, "res5a_branch2c/weights": 46, "res5a_branch2b/biases": 45, "res5a_branch2b/weights": 45,
              "res5a_branch2a/biases": 44, "res5a_branch2a/weights": 44, "res5a_branch1/biases":  43, "res5a_branch1/weights":  43, "res4f_branch2c/biases": 42, "res4f_branch2c/weights": 42,
              "res4f_branch2b/biases": 41, "res4f_branch2b/weights": 41, "res4f_branch2a/biases": 40, "res4f_branch2a/weights": 40, "res4e_branch2c/biases": 39, "res4e_branch2c/weights": 39,
              "res4e_branch2b/biases": 38, "res4e_branch2b/weights": 38, "res4e_branch2a/biases": 37, "res4e_branch2a/weights": 37, "res4d_branch2c/biases": 36, "res4d_branch2c/weights": 36,
              "res4d_branch2b/biases": 35, "res4d_branch2b/weights": 35, "res4d_branch2a/biases": 34, "res4d_branch2a/weights": 34, "res4c_branch2c/biases": 33, "res4c_branch2c/weights": 33,
              "res4c_branch2b/biases": 32, "res4c_branch2b/weights": 32, "res4c_branch2a/biases": 31, "res4c_branch2a/weights": 31, "res4b_branch2c/biases": 30, "res4b_branch2c/weights": 30,
              "res4b_branch2b/biases": 29, "res4b_branch2b/weights": 29, "res4b_branch2a/biases": 28, "res4b_branch2a/weights": 28, "res4a_branch2c/biases": 27, "res4a_branch2c/weights": 27,
              "res4a_branch2b/biases": 26, "res4a_branch2b/weights": 26, "res4a_branch2a/biases": 25, "res4a_branch2a/weights": 25, "res4a_branch1/biases":  24, "res4a_branch1/weights": 24,
              "res3d_branch2c/biases": 23, "res3d_branch2c/weights": 23, "res3d_branch2b/biases": 22, "res3d_branch2b/weights": 22, "res3d_branch2a/biases": 21, "res3d_branch2a/weights": 21,
              "res3c_branch2c/biases": 20, "res3c_branch2c/weights": 20, "res3c_branch2b/biases": 19, "res3c_branch2b/weights": 19, "res3c_branch2a/biases": 18, "res3c_branch2a/weights": 18,
              "res3b_branch2c/biases": 17, "res3b_branch2c/weights": 17, "res3b_branch2b/biases": 16, "res3b_branch2b/weights": 16, "res3b_branch2a/biases": 15, "res3b_branch2a/weights": 15,
              "res3a_branch2c/biases": 14, "res3a_branch2c/weights": 14, "res3a_branch2b/biases": 13, "res3a_branch2b/weights": 13, "res3a_branch2a/biases": 12, "res3a_branch2a/weights": 12,
              "res3a_branch1/biases":  11, "res3a_branch1/weights":  11, "res2c_branch2c/biases": 10, "res2c_branch2c/weights": 10, "res2c_branch2b/biases": 9,  "res2c_branch2b/weights": 9,
              "res2c_branch2a/biases": 8,  "res2c_branch2a/weights": 8,  "res2b_branch2c/biases": 7,  "res2b_branch2c/weights": 7,  "res2b_branch2b/biases": 6,  "res2b_branch2b/weights": 6,
              "res2b_branch2a/biases": 5,  "res2b_branch2a/weights": 5,  "res2a_branch2c/biases": 4,  "res2a_branch2c/weights": 4,  "res2a_branch2b/biases": 3,  "res2a_branch2b/weights": 3,
              "res2a_branch2a/biases": 2,  "res2a_branch2a/weights": 2,  "res2a_branch1/biases":  1,  "res2a_branch1/weights":  1,  "conv1/biases": 0, "conv1/weights": 0}


def weight_quantize(weights, params):
    """
    quantize one float32 to uint8
    :param weights:  numpy float32
    :param layer:  0, 1, 2
    :return: int8
    """

    weights[weights > params['th_params']] = params['th_params']
    weights[weights < -1*params['th_params']] = -1 * params['th_params']
    weights_int8 = weights / params['sf_params']
    weights_int8 = weights_int8.astype(np.int8)

    return weights_int8


def bias_quantize(biases, params):
    """
    quantize one float32 to uint
    :param biases:  numpy float32
    :param layer:  0, 1, 2
    :return: int
    """
    biases_int = biases / (params['sf_params'] * params['sf_layer_in'])
    biases_int = biases_int.astype(np.int32)

    return biases_int


def generate_filter(params, weights, biases):
    filters = []
    for i in range(len(weights)):
        # quantization param
        in_bytes = np.zeros((64,), dtype=np.uint8)
        value = list(layer_name.keys())[list(layer_name.values()).index(i)]
        opname = value.split('/')[0]+"/Conv2D"
        num = np.round(params[opname]['sf_layer_out'] / (params[opname]['sf_params'] * params[opname]['sf_layer_in']))
        in_bytes[63] = [i for i in range(9) if num < pow(2,i)+1][0]
        filters.append(in_bytes)

        # bias
        bias = biases[i]
        bias = np.array(bias, dtype=np.uint32)
        for j in range(int((bias.shape[0]+15)/16)):
            in_bytes = np.zeros((16,), dtype=np.uint32)
            for k in range(16):
                if k < bias.shape[0] - j*16:
                    in_bytes[k] = bias[j * 16 + 16 - k - 1]

                else:
                    in_bytes[k] = 0
            filters.append(in_bytes)
            #print(in_bytes)

        # weight
        weight = weights[i]
        weight = np.array(weight, dtype=np.uint8) # h, w, c_in, c_out
        weight = weight.swapaxes(3, 0)  # c_out, w, c_in, h
        weight = weight.swapaxes(3, 2)  # c_out, w, h, c_in
        weight = weight.swapaxes(2, 1)  # c_out, h, w, c_in
        shape = weight.shape # c_out,h,w,c_in
        for j in range(shape[0]):
            for k in range(int((shape[3]+63)/64)):
                for l in range(shape[1]):
                    for m in range(shape[2]):
                        in_bytes = np.zeros((64,), dtype=np.uint8)
                        for n in range(64):
                            if n < shape[3]-k*64:
                                in_bytes[n] = weight[j][l][m][k*64+n]
                            else:
                                in_bytes[n] = 0
                    filters.append(in_bytes)

    return filters

def quantization_tensorflow(deploy_model, params):

    weights_int8data = {}
    biases_int32data = {}

    tf.reset_default_graph()
    output_graph_path = deploy_model
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        output_graph_def = tf.GraphDef()

        graph = tf.get_default_graph()
        with gfile.FastGFile(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

            operate_value = {}
            operates = [op for op in sess.graph.get_operations() if op.type == "Const"]

            for op in operates:
                operate_value[op.name] = sess.run(op.outputs[0])
                #print(operate_value[op.name].shape)
                if 'fc1000' in op.name:
                    continue

                if 'weights' in op.name:
                    layer_index = layer_name[op.name]
                    int8data = weight_quantize(operate_value[op.name], params[op.name[:-7]+'Conv2D'])
                    weights_int8data[layer_index] = int8data
                elif 'biases' in op.name:
                    layer_index = layer_name[op.name]
                    int32data = bias_quantize(operate_value[op.name], params[op.name[:-6]+'Conv2D'])
                    biases_int32data[layer_index] = int32data
    return weights_int8data, biases_int32data


class quantize_instrument_func:
    _lock = threading.Lock()

    def __init__(self, graph, opnode, inname, quant_param, quant_layers, bitwidth):
        self._graph = graph
        self._opnode = opnode
        self._opname = opnode.name
        self._inname = inname
        self._quant_layers = quant_layers
        self._quant_param = quant_param
        self._bitwidth = bitwidth

    def __call__(self, x):
        if self._opnode.type == 'Conv2D':
            newx = self.quantizeConv2D(x)
        elif self._opnode.type == 'ConcatV2':
            newx = self.quantizeConcat(x)
        else:
            newx = x
        return newx


    def quantizeConv2D(self, x):
        xmin = np.min(x)
        xmax = np.max(x)
        xstd = np.std(x)
        self._quant_layers.add((self._opname, 'unknown'))
        if self._inname == 'image_out':
            descr = 'Mode: %s Image (h,w) = (%d,%d) Chan = %d' % (self._inname, x.shape[1], x.shape[2], x.shape[3])
            self._quant_param.bw_layer_in[self._opname] = self._bitwidth
            self._quant_param.bw_layer_out[self._opname] = self._bitwidth
            threshold = ThresholdLayerOutputs(x, self._bitwidth)
            self._quant_param.th_layer_out[self._opname] = threshold
        elif self._inname == 'image_in':
            descr = 'Mode: %s Image (h,w) = (%d,%d) Chan = %d' % (self._inname, x.shape[1], x.shape[2], x.shape[3])
            self._quant_param.bw_layer_in[self._opname] = self._bitwidth
            self._quant_param.bw_layer_out[self._opname] = self._bitwidth
            threshold = ThresholdLayerOutputs(x, self._bitwidth)
            self._quant_param.th_layer_in[self._opname] = threshold
        elif self._inname == 'weights':
            descr = 'Mode: %s Kernel: %dx%d inChan = %d outChan = %d' % (self._inname, x.shape[0], x.shape[1], x.shape[2], x.shape[3])
            self._quant_param.bw_params[self._opname] = self._bitwidth
            tdata = x.transpose(3, 2, 0, 1)
            self._quant_param.th_params[self._opname] = ThresholdWeights_per_layer(tdata, self._bitwidth)
        msg = 'Op:' + self._opname + ' ' + descr + ' Min: %f, Max: %f, Stddev: %f' % (xmin, xmax, xstd)
        return x


    def findDriver(self, sinktensor):
        srcop = sinktensor.op
        if srcop.name in self._quant_param.bw_layer_out and srcop.name in self._quant_param.th_layer_out:
            return srcop.name
        else:
            for srctensor in srcop.inputs:
                d = self.findDriver(srctensor)
                if d is not None:
                    return d

            return


    def quantizeConcat(self, x):
        if self._inname == 'image_out':
            threshold = ThresholdLayerOutputs(x, self._bitwidth)
            self._quant_param.th_layer_in[self._opname] = threshold
            self._quant_param.th_layer_out[self._opname] = threshold
            self._quant_param.bw_layer_in[self._opname] = self._bitwidth
            self._quant_param.bw_layer_out[self._opname] = self._bitwidth
            for sinkt in self._opnode.inputs:
                d = self.findDriver(sinkt)
                if d is not None:
                    self._quant_param.bw_layer_out[d] = self._bitwidth
                    self._quant_param.th_layer_out[d] = threshold

        return x


def TFlayerName2QuantizeKey(name):
    origName = name
    try:
        name = name.split('/', 1)[0]
        underscores = [ i for i, ltr in enumerate(name) if ltr == '_' ]
        name_list = list(name)
        if len(underscores) <= 2:
            if 'inception' in name:
                name_list[underscores[1]] = '/'
            else:
                name_list[underscores[0]] = '/'
        elif len(underscores) > 2:
            name_list[underscores[1]] = '/'
        name = ('').join(name_list)
    except:
        name = origName

    return name


class QuantizeTF:
    _quant_layers = None
    _instrumentNodes = None
    _g = None
    _quant_param = None

    def __init__(self, tfgraph, bitwidth, quant_param):
        self._quant_layers = set()
        self._g = tfgraph
        self._quant_param = quant_param #QuantParam()
        self._instrumentNodes = []
        convnodes = []
        for op in self._g.get_operations():
            if op.type == 'Conv2D':
                my_func_instr = quantize_instrument_func(self._g, op, 'image_in', self._quant_param, self._quant_layers, bitwidth)
                name = op.name + '_QUANT_IN'
                convnodes.append(tf.py_func(my_func_instr, [op.inputs[0]], tf.float32, name=name))
                my_func_instr = quantize_instrument_func(self._g, op, 'image_out', self._quant_param, self._quant_layers, bitwidth)
                name = op.name + '_QUANT_OUT'
                convnodes.append(tf.py_func(my_func_instr, [op.outputs[0]], tf.float32, name=name))
                tf.summary.histogram(name, op.outputs[0])
                my_func_instr = quantize_instrument_func(self._g, op, 'weights', self._quant_param, self._quant_layers, bitwidth)
                name = op.name + '_QUANT_WEIGHT'
                convnodes.append(tf.py_func(my_func_instr, [op.inputs[1]], tf.float32, name=name))
                tf.summary.histogram(name, op.inputs[1])

        self._instrumentNodes.extend(convnodes)
        with tf.control_dependencies(convnodes):
            for op in self._g.get_operations():
                if op.type == 'ConcatV2':
                    my_func_instr = quantize_instrument_func(self._g, op, 'image_out', self._quant_param, self._quant_layers, bitwidth)
                    name = op.name + '_CONCAT_OUT'
                    self._instrumentNodes.append(tf.py_func(my_func_instr, [op.outputs[0]], tf.float32, name=name))
                    tf.summary.histogram(name, op.outputs[0])

    def getInstrumentNodes(self):
        return self._instrumentNodes

    def writeJson(self, fname):
        self._quant_param.saveToJson(list(self._quant_layers), fname)


class tf_Quantizer:

    def __init__(self, model_file=None, bitwidths=8, quant_params=None, calibration_filenames=None, cal_size=1, img_mean=np.array((104.006989, 116.66877, 122.678917), dtype=np.float32), cal_seed=0, cal_indices=None):
        tf.reset_default_graph()
        with gfile.FastGFile(model_file, 'rb') as (f):
            graphdef = tf.GraphDef()
            graphdef.ParseFromString(f.read())
            _ = tf.import_graph_def(graphdef, name='')
        self.graph = tf.get_default_graph()
        self.bitwidths = bitwidths
        self.calibration_filenames = calibration_filenames
        self.calibration_size = cal_size
        self.img_mean = img_mean
        self.calibration_indices = cal_indices
        self.quant_params = quant_params
        self.calibration_seed = cal_seed

    def preprocessImage(self, fname, imgsize):
        imgdec = tf.image.decode_jpeg(tf.read_file(fname), channels=3)
        batch1out = tf.expand_dims(imgdec, 0)
        resized1 = tf.image.resize_images(batch1out, [imgsize[0], imgsize[1]], tf.image.ResizeMethod.BILINEAR)
        imgred, imggreen, imgblue = tf.unstack(resized1, axis=-1)
        resized2 = tf.stack([imgblue, imggreen, imgred], axis=-1)
        img_mean_const = self.img_mean
        mean_image = tf.subtract(resized2, img_mean_const)
        return mean_image

    def getPlaceholderShape(self):
        ph_shape = None
        for node in self.graph.get_operations():
            if node.type == 'Placeholder':
                shape = node.outputs[0].shape
                return (
                 shape[1].value, shape[2].value, shape[3].value)

        return (None, None, None)
        #return

    def display_results(self, image_paths, probs):
        with open('synset_words.txt', 'rb') as (infile):
            class_labels = list(map(str.strip, infile.readlines()))
        class_indices = np.argmax(probs, axis=1)
        print ('\n{:20} {:30} {}').format('Image', 'Classified As', 'Confidence')
        print '-' * 70
        for img_idx, image_path in enumerate(image_paths):
            img_name = osp.basename(image_path)
            class_name = class_labels[class_indices[img_idx]]
            class_name = class_name[class_name.find(' ') + 1:]
            confidence = round(probs[(img_idx, class_indices[img_idx])] * 100, 2)
            print ('{:20} {:30} {} %').format(img_name, class_name, confidence)

    def quantize(self,  inputName=None, outputName=None):
        img_shape = self.getPlaceholderShape()
        imgs = []
        for index in self.calibration_indices:
            if self.calibration_filenames[index][-5:] == '.JPEG':
                imgs.append( self.calibration_filenames[index])
        print imgs
        prep_inps = []
        with tf.Session() as (sess):
            for img in imgs:
                prep_inps.append(self.preprocessImage(img, img_shape).eval())

        prep_inps = np.concatenate(prep_inps)
        bitwidth = self.bitwidths[0]
        quant_tf = QuantizeTF(tf.get_default_graph(), bitwidth, self.quant_params)
        with tf.Session() as (sess):
            print 'Classifying'
            inputnode = sess.graph.get_operation_by_name(inputName).outputs[0]
            outputnode = sess.graph.get_operation_by_name(outputName).outputs[0]
            results = sess.run([outputnode, quant_tf.getInstrumentNodes()], feed_dict={inputnode: prep_inps})
        #quant_tf.writeJson(self.quantize_config)
        return list(quant_tf._quant_layers)