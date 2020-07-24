# pseudocode templates
# id CONV parents children invalid_wait batch_size conv_type h_shift v_shift h_pad v_pad stride input_width input_height input_channel output_channel input_addr filter_addr quant_addr output_addr width_shift width_size height_shift height_size channel_shift channel_size
# id ACTIVE parents children invalid_wait batch_size active_tupe input_width input_height input_channel output_width output_height output_channel input_addr output_addr
# id POOL parents children invalid_wait batch_size pool_type kernel_size stride input_width input_height input_channel input_addr output_addr
# id ELTWISE parents children invalid_wait batch_size input_width input_height input_channel input_addr residual_addr output_addr
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read(os.path.dirname(os.path.abspath(__file__)) + '/../config.ini')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.basicConfig(filename='tmp.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_HW_BATCH = config.getint('config', 'max_hw_batch')
HW_BATCH_SIZE = config.getint('config', 'hw_batch_size')
ADDR_MASK48 = 0xFFFFFFFFFFFF
REG_INUSED = [False for _ in range(64)]
FREE_REG_NUM = 64
PARAM_REG_HISTORY = []
SET_REG_HISTORY = {}
CHILDREN_HISTORY = {}
INSTR_CODE_DICT = {
    'CONV': 0x000,
    'SPLIT': 0x100,
    'CONCAT': 0x101,
    'ACTIVE': 0x107,
    'ELTWISE': 0x108,
    'MOVH': 0x200,
    'MOVL': 0x201,
    'POOL': 0x400,
    'NOTIFY': 0x301,
    'END': 0x302,
}


# register can be classified as | wait register | param regitster | set regitster |
def _generate_regs_params(instr_name, runid=0, parents=[], children=[], siblings=[], batch_size=4, conv_type=0,
                          active_type=0, pool_type=0, eltwise_type=0, concat_channel=0, input_width=0, input_height=0, input_channel=0, input_mat_width=0,
                          input_mat_height=0, input_vect_width=0, input_vect_height=0, output_width=0, output_height=0,
                          output_channel=0, input_addr=0, filter_addr=0, quant_addr=0, residual_addr=0, output_addr=0,
                          data_addr=0, input_mat_addr=0, input_vect_addr=0, input_bias_addr=0, output_mat_addr=0,
                          h_pad=0, v_pad=0, stride=1, kernel_size=1, input_size=0, output_size=0, width_shift=0,
                          width_size=0, height_shift=0, height_size=0, channel_shift=0, channel_size=0, op='', name='',
                          attrs={}, shape=[], input_shape=[],output_shape=[]):
    regs_params = []
    input_feature_size4 = int(input_size / MAX_HW_BATCH)
    output_feature_size4 = int(output_size / MAX_HW_BATCH)

    invalid_wait = 0
    if len(children) == 0:
        invalid_wait += 2
    min_sibling = min(siblings) if len(siblings) > 0 else runid
    if len(parents) == 0 or runid > min_sibling:
        invalid_wait += 1

    if instr_name in ['CONV']:
        for _ in range(MAX_HW_BATCH):
            reg2 = ((invalid_wait & 0x3) << 56) | ((batch_size & 0xFF) << 48) | (input_addr & ADDR_MASK48)
            reg3 = ((width_shift & 0xFFF) << 48) | (filter_addr & ADDR_MASK48)
            reg4 = ((width_size & 0xFFF) << 48) | (quant_addr & ADDR_MASK48)
            '''
            #before 20191128, add activate into conv
            #reg5 = output_addr & ADDR_MASK48
            #reg6 = ((conv_type & 0x7) << 61) | ((kernel_size & 0xF) << 57) | ((h_pad & 0x7) << 54) | (
            #        (v_pad & 0x7) << 51) | ((stride & 0x7) << 48) | ((input_width & 0xFFF) << 36) | (
            #              (input_height & 0xFFF) << 24) | ((input_channel & 0xFFF) << 12) | (output_channel & 0xFFF)
            '''

            reg5 = ((active_type & 0x7) << 52) | ((conv_type & 0xf) << 48) | (output_addr & ADDR_MASK48)
            reg6 = ((kernel_size & 0xF) << 57) | ((h_pad & 0x7) << 54) | (
                    (v_pad & 0x7) << 51) | ((stride & 0x7) << 48) | ((input_width & 0xFFF) << 36) | (
                           (input_height & 0xFFF) << 24) | ((input_channel & 0xFFF) << 12) | (output_channel & 0xFFF)
            reg7 = ((height_shift & 0xFFF) << 36) | ((height_size & 0xFFF) << 24) | ((channel_shift & 0xFFF) << 12) | (
                    channel_size & 0xFFF)
            input_addr += input_feature_size4
            output_addr += output_feature_size4
            logger.info('generate register parameters %d %s reg2=%#x reg3=%#x reg4=%#x reg5=%#x reg6=%#x reg7=%#x',
                        runid, instr_name, reg2, reg3, reg4, reg5, reg6, reg7)
            regs_params.append([reg2, reg3, reg4, reg5, reg6, reg7])

    elif instr_name in ['ACTIVE', 'ATTACH', 'RESHAPE', 'SPLIT']:
        for _ in range(MAX_HW_BATCH):
            reg2 = ((invalid_wait & 0x3) << 56) | ((batch_size & 0xFF) << 48) | (input_addr & ADDR_MASK48)
            reg3 = ((active_type & 0x7) << 48) | (output_addr & ADDR_MASK48)
            reg4 = ((input_width & 0xFFF) << 48) | ((input_height & 0xFFF) << 36) | ((input_channel & 0xFFF) << 24) | (
                    (width_shift & 0xFFF) << 12) | (width_size & 0xFFF)
            reg5 = ((height_shift & 0xFFF) << 36) | ((height_size & 0xFFF) << 24) | ((channel_shift & 0xFFF) << 12) | (
                    channel_size & 0xFFF)
            reg6 = ((output_width & 0xFFF) << 24) | ((output_height & 0xFFF) << 12) | (output_channel & 0xFFF)
            reg7 = quant_addr & ADDR_MASK48
            logger.info('generate register parameters %d %s reg2=%#x reg3=%#x reg4=%#x reg5=%#x reg6=%#x, reg7=%#x',
                        runid, instr_name, reg2, reg3, reg4, reg5, reg6, reg7)
            input_addr += input_feature_size4
            output_addr += output_feature_size4
            regs_params.append([reg2, reg3, reg4, reg5, reg6, reg7])

    elif instr_name in ['CONCAT']:
        for _ in range(MAX_HW_BATCH):
            reg2 = ((invalid_wait & 0x3) << 56) | ((batch_size & 0xFF) << 48) | (input_addr & ADDR_MASK48)
            reg3 = (output_addr & ADDR_MASK48)
            reg4 = ((concat_channel & 0xFFF) << 48) | ((input_width & 0xFFF) << 36) | ((input_height & 0xFFF) << 24) | ((input_channel & 0xFFF) << 12)
            logger.info('generate register parameters %d %s reg2=%#x reg3=%#x', runid, instr_name, reg2, reg3, reg4)
            input_addr += input_feature_size4
            output_addr += input_feature_size4 + output_feature_size4
            regs_params.append([reg2, reg3, reg4])

    elif instr_name in ['ELTWISE']:
        for _ in range(MAX_HW_BATCH):
            reg2 = ((invalid_wait & 0x3) << 56) | ((batch_size & 0xFF) << 48) | (input_addr & ADDR_MASK48)
            reg3 = ((input_width & 0xFFF) << 48) | (quant_addr & ADDR_MASK48)
            reg4 = ((eltwise_type & 0x3) << 60) | ((input_height & 0xFFF) << 48) | (residual_addr & ADDR_MASK48)
            reg5 = ((active_type & 0x3) << 60) | ((input_channel & 0xFFF) << 48) | (output_addr & ADDR_MASK48)
            logger.info('generate register parameters %d %s reg2=%#x reg3=%#x reg4=%#x reg5=%#x',
                        runid, instr_name, reg2, reg3, reg4, reg5)
            input_addr += input_feature_size4
            residual_addr += input_feature_size4
            output_addr += output_feature_size4
            regs_params.append([reg2, reg3, reg4, reg5])

    elif instr_name in ['POOL']:
        for _ in range(MAX_HW_BATCH):
            reg2 = ((invalid_wait & 0x3) << 56) | ((batch_size & 0xFF) << 48) | (input_addr & ADDR_MASK48)
            reg3 = output_addr & ADDR_MASK48
            reg4 = ((input_width & 0xFFF) << 24) | ((input_height & 0xFFF) << 12) | (input_channel & 0xFFF)
            reg5 = ((quant_addr & ADDR_MASK48) << 14) | ((h_pad & 0x7) << 11) | ((v_pad & 0x7) << 8) | (
                    (pool_type & 0x3) << 6) | ((kernel_size & 0x7) << 3) | (stride & 0x7)
            logger.info('generate register parameters %d %s reg2=%#x reg3=%#x reg4=%#x reg5=%#x',
                        runid, instr_name, reg2, reg3, reg4, reg5)
            input_addr += input_feature_size4
            output_addr += output_feature_size4
            regs_params.append([reg2, reg3, reg4, reg5])

    else:
        print('Error: {} not support!'.format(instr_name))
        sys.exit(1)

    return regs_params


def _allocate_physical_regs(runid, instr_name, parents, children, type=0):
    if instr_name in ['CONV', 'ACTIVE', 'ATTACH', 'RESHAPE', 'SPLIT']:
        reg_num = 8
    elif instr_name in ['ELTWISE']:
        reg_num = 7
    elif instr_name in ['POOL', 'CONCAT']:
        reg_num = 6
    else:
        print('Error: {} not support!'.format(instr_name))
        sys.exit(1)

    regs = []
    set_regs = []
    for batchi in range(MAX_HW_BATCH):
        # free param register
        global FREE_REG_NUM
        global PARAM_REG_HISTORY
        if len(PARAM_REG_HISTORY) > 0:
            for i in PARAM_REG_HISTORY:
                REG_INUSED[i] = False
                FREE_REG_NUM += 1

        # allocte param register & set register
        if FREE_REG_NUM < reg_num - 1:
            print('Error: {} registers valid, but {} registers needed by {}'.format(FREE_REG_NUM, reg_num, instr_name))
            sys.exit(1)

        batchi_regs = [0 for _ in range(reg_num)]
        num = 1 if instr_name not in ['ELTWISE', 'CONCAT'] else 2
        regid = 0
        needed_reg_num = reg_num - 1 if len(children) == 0 else reg_num
        while num < needed_reg_num:
            if not REG_INUSED[regid]:
                batchi_regs[num] = regid
                REG_INUSED[regid] = True
                FREE_REG_NUM -= 1
                num += 1
            regid += 1

        # set wait register
        if len(parents) > 0:
            # max_parent_id = max(parents)
            for i, parent_id in enumerate(parents):
                if parent_id in SET_REG_HISTORY.keys() and len(CHILDREN_HISTORY[parent_id]):
                    wait_regid = SET_REG_HISTORY[parent_id][batchi]
                    batchi_regs[i] = wait_regid

            if len(parents) > 1 and type == 1:
                max_parent_id = max(parents)
                for i, parent_id in enumerate(parents):
                    if parent_id in SET_REG_HISTORY.keys() and len(CHILDREN_HISTORY[parent_id]):
                        wait_regid = SET_REG_HISTORY[max_parent_id][batchi]
                        batchi_regs[i] = wait_regid

            if len(parents) > 0:
                max_parent_id = max(parents)
                if max_parent_id in SET_REG_HISTORY.keys() and len(CHILDREN_HISTORY[max_parent_id]):
                    # free parents' set regitster
                    max_brother_id = max(CHILDREN_HISTORY[max_parent_id])
                    if runid == max_brother_id:
                        for parent_id in parents:
                            parent_regid = SET_REG_HISTORY[parent_id][batchi]
                            REG_INUSED[parent_regid] = False
                            FREE_REG_NUM += 1
                            if batchi == MAX_HW_BATCH - 1:
                                del SET_REG_HISTORY[parent_id]
                                del CHILDREN_HISTORY[parent_id]
        '''
        if len(parents) > 0:
            max_parent_id = max(parents)
            if max_parent_id in SET_REG_HISTORY.keys() and len(CHILDREN_HISTORY[max_parent_id]):
                wait_regid = SET_REG_HISTORY[max_parent_id][batchi]
                batchi_regs[0] = wait_regid

                # free parents' set regitster
                max_brother_id = max(CHILDREN_HISTORY[max_parent_id])
                if runid == max_brother_id:
                    for parent_id in parents:
                        parent_regid = SET_REG_HISTORY[parent_id][batchi]
                        REG_INUSED[parent_regid] = False
                        FREE_REG_NUM += 1
                        if batchi == MAX_HW_BATCH - 1:
                            del SET_REG_HISTORY[parent_id]
                            del CHILDREN_HISTORY[parent_id]
        '''

        PARAM_REG_HISTORY = batchi_regs[1:-1] if instr_name not in ['ELTWISE', 'CONCAT'] else batchi_regs[2:-1]
        set_regs.append(batchi_regs[-1])
        regs.append(batchi_regs)

        loginfo = ' '.join(['%d' for _ in range(reg_num)])
        logger.info('allocate registers %d %s ' + loginfo, runid, instr_name, *batchi_regs)

    SET_REG_HISTORY[runid] = set_regs
    CHILDREN_HISTORY[runid] = children

    return regs


def _pseudo_to_assems(instr_name, regs, reg_params):
    assems = []
    reg_param_num = len(reg_params)
    if instr_name in ['ELTWISE', 'CONCAT']:
        for i in range(reg_param_num):
            assem = ['MOVH', regs[i + 2], reg_params[i] >> 32]
            assems.append(assem)
            assem = ['MOVL', regs[i + 2], reg_params[i] & 0xFFFFFFFF]
            assems.append(assem)
    else:
        for i in range(reg_param_num):
            assem = ['MOVH', regs[i + 1], reg_params[i] >> 32]
            assems.append(assem)
            assem = ['MOVL', regs[i + 1], reg_params[i] & 0xFFFFFFFF]
            assems.append(assem)

    assems.append([instr_name] + regs)
    return assems


def _assem_to_machine_code(assem, batchi):
    instr_name = assem[0]
    creg = batchi + 1
    instr_code = INSTR_CODE_DICT[instr_name]
    machine_code = (creg << 60) | (instr_code << 48)
    if instr_name in ['MOVH', 'MOVL']:
        machine_code += (assem[1] << 42) | assem[2]
    elif instr_name in ['CONV', 'ELTWISE', 'POOL', 'ACTIVE', 'ATTACH', 'CONCAT', 'RESHAPE', 'SPLIT']:
        reg_num = len(assem) - 1
        for i in range(1, reg_num + 1):
            machine_code += (assem[i] << (6 * (8 - i)))
    else:
        print('Error: {} not support!'.format(instr_name))
        sys.exit(1)
    return machine_code


def _pseudo_to_machine_code(pseudo):
    print(pseudo)
    regs_params = _generate_regs_params(**pseudo)
    type = (pseudo['eltwise_type'] & 0x2 >> 1) if pseudo['instr_name'] in ['ELTWISE'] else 0
    regs = _allocate_physical_regs(pseudo['runid'], pseudo['instr_name'], pseudo['parents'], pseudo['children'], type)
    machine_codes = []
    for batchi in range(MAX_HW_BATCH):
        batchi_reg_params = regs_params[batchi]
        batchi_regs = regs[batchi]
        assems = _pseudo_to_assems(pseudo['instr_name'], batchi_regs, batchi_reg_params)
        for assem in assems:
            machine_code = _assem_to_machine_code(assem, batchi)
            machine_codes.append(machine_code)
            logger.info('machine code %s %#018x', assem[0], machine_code)
    return machine_codes


def _int_to_bytes_little_endian(val, byte_width):
    if not isinstance(val, int):
        print('Error: {} must be int'.format(val))
        sys.exit(1)
    byte_list = []
    for i in range(byte_width):
        byte_list.append(val & 0xFF)
        val >>= 8
    return bytes(byte_list)


def _generate_start_byte_codes():
    machine_codes = []
    for i in range(1, MAX_HW_BATCH + 1):
        # CMPGEC CRegi R0 i
        machine_code = (0x213 << 48) | (i << 42) | i
        machine_codes.append(machine_code)
        logger.info('start machine code CMPGEC CReg%d R0 %#018x', i, machine_code)

    byte_codes = []
    for machine_code in machine_codes:
        byte_code = _int_to_bytes_little_endian(machine_code, 8)
        byte_codes.append(byte_code)
    return byte_codes


def _generate_end_byte_codes():
    logger.info('end machine code NOTIFY 0x0301000000000000')
    logger.info('end machine code END 0x0302000000000000')
    return [b'\x00\x00\x00\x00\x00\x00\x01\x03', b'\x00\x00\x00\x00\x00\x00\x02\x03']


def assemble(graph):
    """convert compute graph to instruction stream"""
    instr_streams = []
    instr_streams.extend(_generate_start_byte_codes())

    for node in graph:
        # TODO
        if node['op'] in ['squeeze', 'reshape', 'flatten', 'softmax', 'dense', 'transpose']:
            continue
        machine_codes = _pseudo_to_machine_code(node)
        for machine_code in machine_codes:
            byte_code = _int_to_bytes_little_endian(machine_code, 8)
            instr_streams.append(byte_code)

    instr_streams.extend(_generate_end_byte_codes())

    return np.frombuffer(b"".join(instr_streams), dtype=np.uint8)


