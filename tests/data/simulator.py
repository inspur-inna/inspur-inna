import numpy as np
import quantizer

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

QUANT_MODE = 1

def pool2d(input, kernel=3, stride=2, padding=0, pooltype='max', poolmode='valid'):
    assert pooltype in ['max', 'avg']
    assert poolmode in ['valid', 'full']
    channel, height, width = input.shape

    # padding
    if padding > 0:
        pad_w = np.zeros((channel, height, padding), dtype=np.uint8)
        inputp = np.concatenate((pad_w, input, pad_w), axis=2)
        pad_h = np.zeros((channel, padding, width + 2*padding), dtype=np.uint8)
        inputp = np.concatenate((pad_h, inputp, pad_h), axis=1)
    else:
        inputp = input

    # mode
    height = height + 2*padding
    width = width + 2*padding
    if poolmode == 'full':
        modh = (height - kernel) % stride
        modw = (width - kernel) % stride
        if modh != 0:
            pad_w = np.zeros((channel, height, stride - modh), dtype=np.uint8)
            inputp = np.concatenate((inputp, pad_w), axis=2)
            width += stride - modh
        if modw != 0:
            pad_h = np.zeros((channel, stride - modw, width), dtype=np.uint8)
            inputp = np.concatenate((inputp, pad_h), axis=1)
            height += stride - modw

    oheight = int((height - kernel) / stride + 1)
    owidth = int((width - kernel) / stride + 1)

    # pooling
    output = np.zeros((channel, oheight, owidth), dtype=np.uint8)
    oy = ox = 0
    for y in range(0, height, stride):
        if y + kernel > height:
            continue
        for x in range(0, width, stride):
            if x + kernel > width:
                continue
            pool_data = inputp[:, y:y+kernel, x:x+kernel]
            pool_data = quantizer.anti_quantize(pool_data)
            if pooltype == 'max':
                out_data = np.max(pool_data.reshape(channel, -1), axis=1).astype(np.float32)
            else:
                out_data = np.average(pool_data.reshape(channel, -1), axis=1).astype(np.float32)
            output[:, oy, ox] = quantizer.quantize(out_data, QUANT_MODE)
            ox += 1
            if ox >= owidth:
                ox = 0
                oy += 1

    return output.astype(np.uint8)

def relu(input):
    shape = input.shape
    output = quantizer.anti_quantize(input)
    output = output.reshape(-1)
    for i, elem in enumerate(output):
        if elem < 0:
            output[i] = 0
    output = quantizer.quantize(output, QUANT_MODE)
    return output.reshape(*shape)

def elemwize_add(input0, input1):
    input0 = quantizer.anti_quantize(input0)
    input1 = quantizer.anti_quantize(input1)
    output = np.add(input0, input1)
    output = quantizer.quantize(output, QUANT_MODE)
    return output

def feature_reshape(data, layout='CHW'):
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
        outdata[oci, :, :, 0 : cnum] = data[:, :, start : end]
    return outdata


if __name__ == '__main__':
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    args = parser.parse_args()

    quantizer.generate_quant_binfile(BASE_DIR + 'quant.bin', QUANT_MODE)

    if args.mode == 'pool':
        with open(BASE_DIR + 'pool.json', 'r') as f:
            instr = json.load(f)[0]
        c, w, h = (instr['input_channel'], instr['input_height'], instr['input_width'])
        kernel, pad, stride= (instr['kernel_size'], 0, instr['stride'])
        pooltype = 'max' if instr['pool_type'] & 0x2 == 0 else 'avg'
        infile_name = BASE_DIR + 'pool/pool_input.bin'
        outfile_name = BASE_DIR + 'pool/' + pooltype + '_pool_output.bin'
        input_data = np.arange(c * w * h).reshape(c, w, h).astype(np.uint8)
        output_data = pool2d(input_data, kernel=kernel, stride=stride, padding=pad, pooltype=pooltype, poolmode='full')
        # print(output_data.shape)
        np.set_printoptions(formatter={'int': hex})
        indata = feature_reshape(input_data)
        outdata = feature_reshape(output_data)
        with open(infile_name, 'wb') as f:
            f.write(indata.tobytes())
        with open(outfile_name, 'wb') as f:
            f.write(outdata.tobytes())

    elif args.mode == 'active':
        with open(BASE_DIR + 'active.json', 'r') as f:
            instr = json.load(f)[0]
        c, w, h = (instr['input_channel'], instr['input_height'], instr['input_width'])
        infile_name = BASE_DIR + 'active/active_input.bin'
        outfile_name = BASE_DIR + 'active/active_output.bin'
        input_data = np.arange(c * w * h).reshape(c, w, h).astype(np.uint8)
#        input_data = np.ones(c * w * h).reshape(c, w, h).astype(np.uint8) * -2
        output_data = relu(input_data)
        np.set_printoptions(formatter={'int': hex})
        indata = feature_reshape(input_data)
        outdata = feature_reshape(output_data)
        with open(infile_name, 'wb') as f:
            f.write(indata.tobytes())
        with open(outfile_name, 'wb') as f:
            f.write(outdata.tobytes())

    elif args.mode == 'eltwise':
        with open(BASE_DIR + 'eltwise.json', 'r') as f:
            instr = json.load(f)[0]
        c, w, h = (instr['input_channel'], instr['input_height'], instr['input_width'])
        infile_name = BASE_DIR + 'eltwise/eltwise_input.bin'
        outfile_name = BASE_DIR + 'eltwise/eltwise_output.bin'
        input_data = np.arange(c * w * h).reshape(c, w, h).astype(np.uint8)
        output_data = elemwize_add(input_data, input_data)
        # print(output_data.shape)
        np.set_printoptions(formatter={'int': hex})
        indata = feature_reshape(input_data)
        outdata = feature_reshape(output_data)
        with open(infile_name, 'wb') as f:
            f.write(indata.tobytes())
        with open(outfile_name, 'wb') as f:
            f.write(outdata.tobytes())
