from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import configparser
import numpy as np
import math
from inna.runtime import _runtime

import argparse
import pdb

def check_equal(tname, test, real):
    """
    Parameters
    ----------
    test : np.NDArray
        test array

    real : np.NDArray
        real array

    Return
    ------
    ret : bool
        whether two array is equal
    """
    return _runtime.CheckEqual(tname, test, real)


config = configparser.ConfigParser()
config.read(os.path.dirname(os.path.abspath(__file__)) + '/../config.ini')
HW_BATCH_SIZE = config.getint('config', 'hw_batch_size')
MAX_HW_BATCH = config.getint('config', 'max_hw_batch')
MAX_BATCH_SIZE = MAX_HW_BATCH * HW_BATCH_SIZE


def _softmax(x):
    assert x.ndim == 2
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1)


def _fullconv(x, weights, weights_addr, quant, quant_addr, class_num):
    #pdb.set_trace()
    x = x.astype(np.int8)
    np_weight = weights[weights_addr:]
    np_weight = np_weight.astype(np.int8)
    weight_size = np_weight.shape[0]

    np_quant = quant[quant_addr:]
    quant_table = np_quant[0]
    np_quant_int32 = np.fromstring(np_quant[64:].tostring(), dtype=np.int32)

    np_quant_int32_sort = np_quant_int32[:class_num]
    np_weight = np_weight.reshape([class_num, int(weight_size/class_num)])

    x.tofile("x.bin")
    np_weight.tofile("weight.bin")
    np_quant.tofile("bias.bin")

    np_weight = np_weight.astype(np.int32)
    x = x.astype(np.int32)
    y = np.zeros((x.shape[0], class_num), dtype=np.float)

    for k in range(x.shape[0]):
        temp = np.dot(np_weight, x[k])
        temp = temp + np_quant_int32_sort
        temp = temp / (2 ** quant_table)
        y[k] = temp
    #print(" fullconv y=")
    #print(y)
    return y


def _feature_reshape(data, layout='CHW'):
    assert layout in ['CHW', 'HWC']
    if layout == 'CHW':
        data = data.transpose((1, 2, 0))
    h, w, c = data.shape
    oc = int((c + 63) / 64)
    outdata = np.zeros((oc, h, w, 64), dtype=np.uint8)
    for oci in range(oc):
        start = oci * 64
        end = (oci + 1) * 64
        if end > c:
            end = c
        cnum = end - start
        #outdata[oci, :, :, 0: cnum] = data[:, :, start: end]

        for i in range(cnum):
            outdata[oci, :, :, 63-i] = data[:, :, i]

    return outdata


def _load_images(img_dir, batch_size):
    """ load images from img_dir to batchs split by batch_size

    Parameters
    ----------
    img_dir : str
        directory contain JPEG files

    batch_size : int
        images numbers processed per run

    Returns
    -------
    batchi : generator
        current batch

    batch_imgs : generator
        current batch images
    """
    img_dir = os.path.abspath(img_dir) + '/'
    total_imgs = int(os.popen('ls ' + img_dir + '/*.JPEG | wc -l').read())
    cur_img = batchi = 0
    imgs = []
    for filename in os.listdir(img_dir):
        if filename.endswith('.JPEG'):
            img = cv2.imread(img_dir + filename)
            img = cv2.resize(img, (224, 224)) #, interpolation=cv2.INTER_CUBIC)
            #print(img.shape)
            img = img.astype(np.float)
            img[..., 0] = (img[..., 0] - 103.939) / 2.0
            img[..., 1] = (img[..., 1] - 116.779) / 2.0
            img[..., 2] = (img[..., 2] - 123.68) / 2.0

            img = _feature_reshape(img, 'HWC')
            imgs.append(img)
            cur_img += 1
            if cur_img % batch_size == 0 or cur_img == total_imgs:
                batchi += 1
                batch_imgs = np.array(imgs)
                imgs = []
                yield batchi, batch_imgs


def _aligned64(arr):
    arr = arr.reshape(-1)
    mod = arr.size % 64
    if mod != 0:
        pad_data = np.zeros(64 - mod, dtype=arr.dtype)
        arr = np.concatenate((arr, pad_data), axis=0)
    return arr


def _get_addrs():
    instrs_addr = int(config.get('config', 'instrs_addr'), 16)
    assert instrs_addr % 64 == 0
    weights_addr = int(config.get('config', 'weights_addr'), 16)
    assert weights_addr % 64 == 0
    quant_addr = int(config.get('config', 'quant_addr'), 16)
    assert quant_addr % 64 == 0
    return instrs_addr, weights_addr, quant_addr


def _create_model_datainfo(instrs, weights, quant_table):
    instrs_addr, weights_addr, quant_addr = _get_addrs()
    instrs = _aligned64(instrs)
    weights = _aligned64(weights)
    quant_table = _aligned64(quant_table)
    return [instrs_addr, instrs, weights_addr, weights, quant_addr, quant_table]


class INNARuntime(_runtime.INNARuntime):
    def __init__(self, instrs, weights, quant_table):
        """
        instrs : np.NDArray
            instruction stream represented by 1D array

        weights : np.NDArray
            weight array after quantize and data reshape

        quant_table : np.NDArray
            quantize table and quantize parameters
        """
        super().__init__()
        self._ifeature_addr = int(config.get('config', 'input_feature_addr'), 16)
        self._ofeature_addr = int(config.get('config', 'output_feature_addr'), 16)
        self._ofeature_size = config.getint('config', 'per_output_feature_size')
        self._fullconv_weight_addr = int(config.get('config', 'fullconv_weight_addr'), 16)
        self._fullconv_quant_addr = int(config.get('config', 'fullconv_quant_addr'), 16)
        self._class_num = config.getint('config', 'class_num')
        self._weights = weights
        self._quant = quant_table
        self._model = _create_model_datainfo(instrs, weights, quant_table)
        self.ReloadNNModel(*self._model)

    def run(self, img_dir, batch_size):
        """ start reference process

        Parameters
        ----------
        img_dir : str

        batch_size : int

        Returns
        -------
        ret : np.NDArray
            result of neural network model
        """
        batchs = _load_images(img_dir, batch_size)
        outputs = np.zeros((1, self._class_num), dtype=np.uint8)
        for batchi, batch_imgs in batchs:
            print(batchi, batch_imgs.shape)
            hwalign = batch_imgs.shape[0] % HW_BATCH_SIZE
            if hwalign > 0:
                pad_imgs = np.zeros((HW_BATCH_SIZE - hwalign, batch_imgs.shape[1:]), dtype=np.uint8)
                batch_imgs = np.concatenate((batch_imgs, pad_imgs), axis=0)
            for i in range(math.ceil(batch_imgs.shape[0] / MAX_BATCH_SIZE)):
                max_img_range = min((i + 1) * MAX_BATCH_SIZE, batch_imgs.shape[0])
                hw_batch_imgs = batch_imgs[i * MAX_BATCH_SIZE: max_img_range]
                print(hw_batch_imgs.shape, int(hw_batch_imgs.shape[0] / HW_BATCH_SIZE))
                hw_batch_imgs.tofile("input_data")
                self.SetExtendHWBatch(int(hw_batch_imgs.shape[0] / HW_BATCH_SIZE))
                self.SetInputFeatures(self._ifeature_addr, hw_batch_imgs.reshape(-1))
                self.Run()
                self.Wait()
                #####
                '''
                ofeatures_addr = 0x1000000
                ofeatures_size = 0x1000000
                for j in range(119):
                    ofeatures_addr = ofeatures_addr + 0x1000000
                    if ofeatures_addr == 0xc000000:
                        ofeatures_addr = ofeatures_addr + 0x1000000
                    ofeatures = self.GetOutputFeatures(ofeatures_addr,hw_batch_imgs.shape[0] * ofeatures_size)
                    filename = os.path.dirname(os.path.abspath(__file__)) + '/mid_output/runid_' + str(j)
                    with open(filename, 'wb') as fp:
                        fp.write(ofeatures.tostring())
                '''
                #####

                ofeatures = self.GetOutputFeatures(self._ofeature_addr,
                                                   hw_batch_imgs.shape[0] * self._ofeature_size)

                ofeatures = ofeatures.reshape(hw_batch_imgs.shape[0], -1)
                #print(ofeatures)
                #pdb.set_trace()
                ofeatures = _fullconv(ofeatures, self._weights, self._fullconv_weight_addr, self._quant, self._fullconv_quant_addr, self._class_num)
                ofeatures = _softmax(ofeatures)
                outputs = np.row_stack((outputs, ofeatures))
        return outputs[1:]


def create(instrs, weights, quant_table):
    """ create runtime

    Parameters
    ----------
    instrs : np.NDArray
        instruction stream represented by 1D array

    weights : np.NDArray
        weight array after quantize and data reshape

    quant_table : np.NDArray
        quantize table and quantize parameters

    Returns
    -------
    ret : inna.runtime.Runtime
        A Runtime class
    """
    return INNARuntime(instrs, weights, quant_table)


if __name__ == '__main__':
    #from inna import runtime
    #aa = np.array([124, 125], dtype=np.uint8)
    #runtime = runtime.create(aa, aa, aa)
    #runtime.run('images', 10)


    parser = argparse.ArgumentParser()
    parser.add_argument("path", default='~/resnet50_param/', help="input file path")
    args = parser.parse_args()

    basepath = args.path
    instrs = np.fromfile(basepath+'instruction_conv.bin', dtype=np.uint8)
    filters = np.fromfile(basepath + 'filter.bin', dtype=np.uint8)
    quant_table = np.fromfile(basepath + 'quant.bin', dtype=np.uint8)
    with open(basepath+'val.txt', 'r') as fp:
        vtable = fp.read().split('\n')
    vtable = [int(data.split(' ')[1]) for data in vtable if data!='']

    #runtime = runtime.create(instrs, filters, quant_table)
    runtime = create(instrs, filters, quant_table)
    output = runtime.run('images', 1)

    top1 = 0
    top5 = 0
    for i in range(int(output.shape[0])):
        list_a = output[i].tolist()
        max_v = max(list_a)
        max_v_index = list_a.index(max(list_a))
        print(i, end=" ")
        if vtable[i] == max_v_index:
            top1 += 1
            top5 += 1
            print("top1", max_v_index, max_v)
            continue
        else:
            for j in range(4):
                list_a[max_v_index] = 0
                max_v = max(list_a)
                max_v_index = list_a.index(max(list_a))

                if vtable[i] == max_v_index:
                    top5 += 1
                    print("top5", max_v_index, max_v)
                    break
        print("")

    print("top1=",top1)
    print("top5=",top5)
