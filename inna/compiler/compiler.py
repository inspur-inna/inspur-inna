from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inna.compiler import frontend
from inna.compiler import converter
from inna.compiler import scheduler
from inna.compiler import assembler

import tvm.relay

class INNACompiler:
    """Compiler Class"""
    def __init__(self, mode='tensorflow'):
        """
        Parameters
        ----------
        mode : str
            can be 'tensorflow', 'keras', 'mxnet', 'onnx'
        """
        self._mode = mode

    def compile(self, graph, shape_dict, layout):
        """ compile neural network model to instruction stream

        Parameters
        ----------
        graph : tf.GraphDef, keras.engine.traing.Model, mxnet.HybridBlock
            compute graph load from deep learning framework

        shape_dict : dict
            input node shape
            key [input node name], value [shape list/tuple]

        layout : str
            feature map laytout
            can be 'NCHW', 'NHWC'

        Returns
        -------
        ret : np.NDArray
            instruction stream represented by np.NDArray
        """
        assert layout in ['NCHW', 'NHWC']

        tvm_graph, params = frontend.to_tvm(graph, shape_dict, layout, self._mode)
        hwgraph = converter.convert(tvm_graph, shape_dict, layout)
        scheduler.schedule(hwgraph)
        instr_streams = assembler.assemble(hwgraph)
        return instr_streams


def create(mode='tensorflow'):
    """ create compiler

    Parameters
    ----------
    mode : str
        can be 'tensorflow', 'keras', 'mxnet', 'onnx'

    Returns
    -------
    ret : inna.compiler.INNACompiler
        A Compiler class
    """
    assert mode in frontend.FRAME_SUPPORTED
    return INNACompiler(mode)


if __name__ == '__main__':
    import inna
    from inna.compiler import applications
    if True:
        graph, shape_dict, layout = applications.resnet.tensorflow.resnet_v1_50()
        compiler = inna.compiler.create('tensorflow')
        instr_streams = compiler.compile(graph, shape_dict, layout)
        outfile_name = 'out/resnet_v1_50_tensorflow.bin'
        with open(outfile_name, 'wb') as f:
            f.write(instr_streams.tobytes())

    if True:
        graph, shape_dict, layout = applications.resnet.keras.resnet_v1_50()
        compiler = inna.compiler.create('keras')
        instr_streams = compiler.compile(graph, shape_dict, layout)
        outfile_name = 'out/resnet_v1_50_keras.bin'
        with open(outfile_name, 'wb') as f:
            f.write(instr_streams.tobytes())
    if True:
        graph, shape_dict, layout = applications.resnet.mxnet.resnet_v1_50()
        compiler = inna.compiler.create('mxnet')
        instr_streams = compiler.compile(graph, shape_dict, layout)
        outfile_name = 'out/resnet_v1_50_mxnet.bin'
        with open(outfile_name, 'wb') as f:
            f.write(instr_streams.tobytes())
    if True:
        graph, shape_dict, layout = applications.resnet.onnx.resnet_v1_50()
        compiler = inna.compiler.create('onnx')
        instr_streams = compiler.compile(graph, shape_dict, layout)
        outfile_name = 'out/resnet_v1_50_tensorflow.bin'
        with open(outfile_name, 'wb') as f:
            f.write(instr_streams.tobytes())
