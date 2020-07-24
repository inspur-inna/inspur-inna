# uncompyle6 version 3.3.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.7.1 (default, Dec 14 2018, 19:28:38) 
# [GCC 7.3.0]
# Embedded file name: /inna/quantize/quantize.py
# Compiled at: 2019-07-20 08:25:54
import numpy as np, os, sys, argparse, math, collections,  google.protobuf.text_format as tfmt, json #,scipy
from quantize_base import *
#from quantize_caffe import *
from quantize_tf import *
import cv2


class Frontend:

    def __init__(self, framework='tensorflow', deploy_model=None, output_json=None, weights=None, calibration_directory=None, calibration_size=None, calibration_seed=None, calibration_indices=None, bitwidths=None, dims=None, transpose=None, channel_swap=None, raw_scale=None, mean_value=None, input_scale=None):
        self.framework = framework
        self.deploy_model = deploy_model
        self.output_json = output_json
        self.weights = weights
        self.calibration_directory = calibration_directory
        self.calibration_size = calibration_size
        self.calibration_seed = calibration_seed
        self.calibration_indices = calibration_indices
        if bitwidths is None:
            self.bitwidths = [
             16, 16, 16]
        else:
            if not isinstance(bitwidths, list):
                self.bitwidths = eval(bitwidths)
            else:
                self.bitwidths = bitwidths
        if dims is not None and not isinstance(dims, list):
            self.dims = eval(dims)
        else:
            self.dims = dims
        if not isinstance(transpose, list):
            self.transpose = eval(transpose)
        else:
            self.transpose = transpose
        if not isinstance(channel_swap, list):
            self.channel_swap = eval(channel_swap)
        else:
            self.channel_swap = channel_swap
        self.raw_scale = raw_scale
        if not isinstance(mean_value, list):
            self.mean_value = eval(mean_value)
        else:
            self.mean_value = mean_value
        self.input_scale = input_scale
        return

    def quantize(self):
        if self.deploy_model is None:
            print 'Error: You must provide deploy model...'
            return
        if self.calibration_directory is None:
            print 'Error: You must provide calibration directory...'
            return
        calibration_filenames = list((os.path.isfile(os.path.join(self.calibration_directory, calibration_filename)) and calibration_filename for calibration_filename in os.listdir(self.calibration_directory)))
        for i in range(0, len(calibration_filenames)):
            calibration_filenames[i] = os.path.join(self.calibration_directory, calibration_filenames[i])

        calibration_size = None
        if self.calibration_size is not None:
            calibration_size = int(self.calibration_size)
            if calibration_size > len(calibration_filenames):
                print 'Error: Requested calibration size is greater than the number of available files in the specified calibration directory\n'
                return
        else:
            calibration_size = len(calibration_filenames)
        calibration_indices = None
        extra_calibration_indices = None
        np.random.seed(self.calibration_seed)
        if self.calibration_indices is not None:
            calibration_indices = np.array([ int(s) for s in eval(self.calibration_indices) ])
            remaining_array = np.setdiff1d(np.arange(0, len(calibration_filenames)), calibration_indices)
            extra_calibration_indices = np.sort(np.random.choice(remaining_array, calibration_size - len(calibration_indices), replace=False))
            calibration_indices = np.sort(np.append(calibration_indices, extra_calibration_indices))
        else:
            calibration_indices = np.sort(np.random.choice(list(range(len(calibration_filenames))), calibration_size, replace=False))
        bitwidths = self.bitwidths
        dims = self.dims
        transpose = self.transpose
        channel_swap = self.channel_swap
        raw_scale = float(self.raw_scale)
        mean_value = None
        if self.mean_value is not None:
            mean_value_list = [ float(s) for s in self.mean_value ]
            mean_value = np.array(mean_value_list)
        input_scale = float(self.input_scale) if self.input_scale else None
        print ('Mean :', mean_value)
        quant_params = QuantParam()
        #framework = 'caffe'
        if self.framework == 'caffe':
            caffeObj = Caffe(self.deploy_model, self.weights)
            caffeObj.declareTransformer(transpose, channel_swap, raw_scale, mean_value, input_scale, dims=None)
            caffeObj.initializeBatch(calibration_size, calibration_filenames, calibration_indices)
            quant_layers = caffeObj.executeCalibration(bitwidths, self.deploy_model, quant_params)
        else:
            if self.framework == 'tensorflow':
                tfobj = tf_Quantizer(self.deploy_model, bitwidths, quant_params, calibration_filenames, calibration_size, mean_value, self.calibration_seed, calibration_indices)
                quant_layers = tfobj.quantize('data', 'prob')
                #pass
            else:
                print 'Unknown framework %s.  Exiting.' % self.framework
                return
        ofile = self.deploy_model.replace('prototxt', 'json')
        if self.output_json is not None:
            ofile = self.output_json
        pathname = os.path.dirname(ofile)
        if pathname is '':
            pass
        else:
            if not os.path.exists(pathname):
                try:
                    os.makedirs(pathname)
                except:
                    print 'Error: The string given for the output_json contains a path that does not exist, and cannot be created.'
                    sys.exit(1)

        quant_params.saveToJson_per_layer(quant_layers, ofile) #lly
        #quant_params.saveToJson(quant_layers, ofile)
        print 'Arguments:'
        print [ (key, val) for key, val in list(self.__dict__.items()) if not key.startswith('__') ]
        return quant_params,quant_layers#True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parameters = [(None, '--framework', str,  'tensorflow', 'store'),
     (
      None, '--deploy_model',
      str,
      '../models/tensorflow/resnet/resnet50_without_bn_test.pb',
      'store'),
     (
      None,
      '--output_json',
      str, './out/filter.bin', 'store'),
    (None, '--weights', str, '../models/resnet50_without_bn_test.pb', 'store'), (None, '--calibration_directory', str, '../runtime/images/', 'store'), (None, '--calibration_size', int, 8, 'store'), (None, '--calibration_seed', int, None, 'store'), (None, '--calibration_indices', str, None, 'store'), (None, '--bitwidths', str, '[8,8,8]', 'store'), (None, '--dims', str, '[3,224,224]', 'store'), (None, '--transpose', str, '[2,0,1]', 'store'), (None, '--channel_swap', str, '[2,1,0]', 'store'), (None, '--raw_scale', float, 255.0, 'store'), (None, '--mean_value', str, '[104.0,117.0,123.0]', 'store'), (None, '--input_scale', float, 1.0, 'store')]
    for x in parameters:
        if x[0] is not None:
            parser.add_argument(x[0], x[1], type=x[2], default=x[3], action=x[4])
        else:
            parser.add_argument(x[1], type=x[2], default=x[3], action=x[4])

    args = parser.parse_args()
    if not (os.path.isfile(args.deploy_model) and os.access(args.deploy_model, os.R_OK)):
        sys.exit('ERROR: Specified prototxt file does not exist or is not readable.')
    if not (os.path.isfile(args.weights) and os.access(args.weights, os.R_OK)):
        sys.exit('ERROR: Specified caffemodel file does not exist or is not readable.')
    if not (os.path.isdir(args.calibration_directory) and os.access(args.calibration_directory, os.R_OK)):
        sys.exit('ERROR: Specified calibration directory does not exist or is not readable.')
    quantizer = Frontend(args.framework, args.deploy_model, args.output_json, args.weights, args.calibration_directory, args.calibration_size, args.calibration_seed, args.calibration_indices, args.bitwidths, args.dims, args.transpose, args.channel_swap, args.raw_scale, args.mean_value, args.input_scale)
    params, layers = quantizer.quantize()
    params_json = params.jsontype(layers)

    weights_int8data, biases_int32data = quantization_tensorflow(args.deploy_model, params_json)

    filters = generate_filter(params_json, weights_int8data, biases_int32data)

    with open(os.path.dirname(os.path.abspath(__file__)) + '/filters.bin', 'wb') as fp:
        for filter in filters:
            fp.write(filter.tobytes())
# okay decompiling quantize.pyc




