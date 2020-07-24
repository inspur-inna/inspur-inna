from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import argparse
from inna.compiler import assembler 

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

parser = argparse.ArgumentParser()
parser.add_argument('--infile')
parser.add_argument('--outfile')
parser.add_argument('--hw_batch', type=int)
parser.add_argument('--max_batch', type=int)
args = parser.parse_args()

assembler.HW_BATCH_SIZE = args.hw_batch
assembler.MAX_HW_BATCH = args.max_batch

with open(BASE_DIR + args.infile, 'r') as f:
    instrs = json.load(f)

for instr in instrs:
    for key, value in instr.items():
        if key in ['input_addr', 'filter_addr', 'quant_addr', 'output_addr', 'residual_addr']:
            instr[key] = int(value, 16)

instr_streams = assembler.assemble(instrs)
with open(BASE_DIR + args.outfile, 'wb') as f:
    f.write(instr_streams.tobytes())
