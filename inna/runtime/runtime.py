from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import configparser
import numpy as np
import math

from inna.runtime import _runtime


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
        outdata[oci, :, :, 0: cnum] = data[:, :, start: end]
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
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
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
        outputs = np.zeros((1, self._ofeature_size), dtype=np.uin8)
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
                self.SetExtendHWBatch(int(hw_batch_imgs.shape[0] / HW_BATCH_SIZE))
                self.SetInputFeatures(self._ifeature_addr, hw_batch_imgs.reshape(-1))
                self.Run()
                self.Wait()
                ofeatures = self.GetOutputFeatures(self._ofeature_addr, 
                        hw_batch_img.shape[0] * self._ofeature_size)
                ofeatures = ofeatures.reshape(hw_batch_imgs.shape[0], -1)
                ofeatures = _softmax(ofeature)
                outputs = np.row_stack(outputs, ofeatures)
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
    from inna import runtime
    aa = np.array([124, 125], dtype=np.uint8)
    runtime = runtime.create(aa, aa, aa)
    runtime.run('images', 10)
