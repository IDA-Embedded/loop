#!/usr/bin/env python

import sys
import tflite
import numpy as np


def calc_mem(path):
    with open(path, "rb") as f:
        buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)

    graph = model.Subgraphs(0)

    _dict_builtin_op_code_to_name = {
        v: k for k, v in tflite.BuiltinOperator.__dict__.items() if type(v) == int
    }

    def print_header():
        print("%-18s | In+Out memory used" % ("OP_NAME"))
        print("------------------------------")

    def print_mem(op_code_builtin, byte_count):
        print(
            "%-18s | %.1f KBs"
            % (_dict_builtin_op_code_to_name[op_code_builtin], byte_count / 1024)
        )

    def print_footer(largest_byte_count):
        print("------------------------------")
        print("Largest memory usage: %.1f KBs" % (largest_byte_count / 1024))

    largest_byte_count = 0
    print_header()
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        op_code_builtin = op_code.BuiltinCode()

        op_opt = op.BuiltinOptions()

        byte_count = 0
        for i in range(op.InputsLength()):
            input = op.Inputs(i)
            byte_count += np.prod(graph.Tensors(input).ShapeAsNumpy())
        for i in range(op.OutputsLength()):
            output = op.Outputs(i)
            byte_count += np.prod(graph.Tensors(output).ShapeAsNumpy())

        print_mem(op_code_builtin, byte_count)
        if byte_count > largest_byte_count:
            largest_byte_count = byte_count

    print_footer(largest_byte_count)


if __name__ == "__main__":
    path = sys.argv[1]
    calc_mem(path)
