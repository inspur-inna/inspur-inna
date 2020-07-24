from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import numpy as np

from inna import runtime
from inna.runtime import _runtime, check_equal

parser = argparse.ArgumentParser()
parser.add_argument('case')
args = parser.parse_args()

if os.path.exists('./generate_data'):
    os.system('g++ generate_data.cc -o generate_data')

if args.case == 'register':
    start_addr = 0x0
    datafile = 'data/register.bin'
    os.system('./generate_data {} 256 0x1234'.format(datafile))
    with open(datafile, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint32)

    runt = _runtime.INNARuntime()
    runt.WriteRegister(start_addr, data)
    rdata = runt.ReadRegisterU32(start_addr, data.size)[0]

    check_equal('Case ' + args.case, rdata, data)
    os.system('rm -f {}'.format(datafile))

elif args.case == 'dma':
    start_addr = 0x0
    datafile = 'data/dma.bin'
    os.system('./generate_data {} 1G 0x1234'.format(datafile))
    with open(datafile, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    runt = _runtime.INNARuntime()
    runt.WriteDMA(start_addr, data)
    rdata = runt.ReadDMA(start_addr, data.size)[0]

    check_equal('Case ' + args.case, rdata, data)
    os.system('rm -f {}'.format(datafile))

elif args.case == 'instr':
    instr_infile = 'data/resnet50_first_5_layers.json'
    instr_outfile = instr_infile.split('.')[0] + '.bin'
    instr_logfile = instr_infile.split('.')[0] + '.log'
    
    # generate instr
    os.system('python ./generate_instr.py --infile {} ' \
        '--outfile {} --hw_batch 1 --max_batch 1 2> {}' \
        .format(instr_infile, instr_outfile, instr_logfile))
    
    with open(instr_outfile, 'rb') as f:
        instrs_data = np.frombuffer(f.read(), dtype=np.uint8)

    runt = _runtime.INNARuntime()
    runt.WriteDMA(0, instrs_data)
    r_instrs_data = runt.ReadDMA(0, instrs_data.size)[0]
    check_equal('instr stream', r_instrs_data, instrs_data)

    runt.SetExtendHWBatch(1)
    runt.Run()
    runt.Wait()

elif args.case in ['pool', 'active', 'eltwise']:
    instr_infile = 'data/{}.json'.format(args.case)
    quant_file = 'data/quant.bin'
    indatafile = 'data/{}/{}_input.bin'.format(args.case, args.case)
    outdatafile = 'data/{}/{}_output.bin'.format(args.case, args.case)
    
    instr_outfile = instr_infile.split('.')[0] + '.bin'
    instr_logfile = instr_infile.split('.')[0] + '.log'

    with open(instr_infile, 'r') as f:
        instr = json.load(f)[0]
    for key, value in instr.items():
        if key in ['input_addr', 'filter_addr', 'quant_addr', 'output_addr', 'residual_addr']:
            instr[key] = int(value, 16)

    if args.case == 'pool':
        pooltype = 'max' if instr['pool_type'] & 0x2 == 0 else 'avg'
        outdatafile = 'data/pool/{}_pool_output.bin'.format(pooltype)

    # generate instr
    os.system('python ./generate_instr.py --infile {} ' \
        '--outfile {} --hw_batch 1 --max_batch 1 2> {}' \
        .format( instr_infile, instr_outfile, instr_logfile))
    
    # generate data
    os.system('python data/simulator.py {}'.format(args.case))

    # start test
    with open(instr_outfile, 'rb') as f:
        instrs_data = np.frombuffer(f.read(), dtype=np.uint8)
    with open(indatafile, 'rb') as f:
        indata = np.frombuffer(f.read(), dtype=np.uint8)
    with open(quant_file, 'rb') as f:
        quant_data = np.frombuffer(f.read(), dtype=np.uint8)
    with open(outdatafile, 'rb') as f:
        outdata = np.frombuffer(f.read(), dtype=np.uint8)

    runt = _runtime.INNARuntime()
    runt.WriteDMA(0, instrs_data)
    r_instrs_data = runt.ReadDMA(0, instrs_data.size)[0]
    check_equal('instr stream', r_instrs_data, instrs_data)

    runt.WriteDMA(instr['input_addr'], indata)
    r_indata = runt.ReadDMA(instr['input_addr'], indata.size)[0]
    check_equal('input feature', r_indata, indata)

    runt.WriteDMA(instr['quant_addr'], quant_data)
    r_quant_data = runt.ReadDMA(instr['quant_addr'], quant_data.size)[0]
    check_equal('quant table', r_quant_data, quant_data)

    if args.case == 'eltwise':
        runt.WriteDMA(instr['residual_addr'], indata)
        r_residual_data = runt.ReadDMA(instr['residual_addr'], indata.size)[0]
        check_equal('residual feature', r_residual_data, indata)

    runt.SetExtendHWBatch(1)
    runt.Run()
    runt.Wait()
    
    r_outdata = runt.ReadDMA(instr['output_addr'], outdata.size)[0]
    check_equal('output feature', r_outdata, outdata)

else:
    print('usage : {} register/dma/instr/active/eltwise/pool'.format(__file__))
