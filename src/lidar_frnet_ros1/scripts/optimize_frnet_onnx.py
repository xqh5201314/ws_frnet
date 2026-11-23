#!/usr/bin/env python3
"""Export the bundled 17-class FRNet with CUDA max pooling by frustum."""

import argparse
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


DOMAIN = "lidar_frnet_ros1"
EXPECTED_SCATTERS = {
    "/voxel_encoder/ScatterElements_2",
    "/backbone/ScatterElements",
    *(f"/backbone/ScatterElements_{i}" for i in range(1, 5)),
}


def attributes(node):
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def optimize(model):
    graph = model.graph
    if {item.name for item in graph.input} != {"points", "coors", "voxel_coors", "inverse_map"}:
        raise ValueError("Expected the bundled FRNet's four input tensors")
    if [item.name for item in graph.output] != ["seg_logit"]:
        raise ValueError("Expected the bundled FRNet's seg_logit output")

    producers = {tensor: node for node in graph.node for tensor in node.output if tensor}
    constants = {tensor.name: tensor for tensor in graph.initializer}
    constants.update({node.output[0]: attributes(node)["value"] for node in graph.node
                      if node.op_type == "Constant" and "value" in attributes(node)})

    reduce = next((node for node in graph.node if node.op_type == "ReduceMax" and
                   list(node.input) == ["inverse_map"] and attributes(node).get("keepdims") == 0),
                  None)
    if reduce is None or reduce.name != "/voxel_encoder/ReduceMax":
        raise ValueError("Could not verify the bundled FRNet's frustum count")
    count_add = next((node for node in graph.node if node.name == "/voxel_encoder/Add"), None)
    if count_add is None or count_add.op_type != "Add" or reduce.output[0] not in count_add.input:
        raise ValueError("Expected the maximum frustum ID plus one")
    addend = next(name for name in count_add.input if name != reduce.output[0])
    constant = numpy_helper.to_array(constants[addend])
    if constant.shape != () or constant.dtype.name != "int64" or constant.item() != 1:
        raise ValueError("Expected a scalar INT64 one")

    replacements = {}
    for node in graph.node:
        if node.name not in EXPECTED_SCATTERS:
            continue
        if node.op_type != "ScatterElements" or attributes(node) != {"axis": 0, "reduction": b"max"}:
            raise ValueError(f"Unexpected pooling operator at {node.name}")
        initial, indices, updates = node.input
        expanded = producers[indices]
        rows = producers[expanded.input[0]]
        if expanded.op_type != "Expand" or rows.op_type != "Unsqueeze" or rows.input[0] != "inverse_map":
            raise ValueError(f"Unexpected inverse_map expansion at {node.name}")
        axes = numpy_helper.to_array(constants[rows.input[1]])
        if axes.tolist() not in ([1], [-1]):
            raise ValueError(f"Expected a row index at {node.name}")
        replacements[node.name] = helper.make_node(
            "FrnetFrustumAtomicMax", [initial, "inverse_map", updates], list(node.output),
            name=node.name, domain=DOMAIN,
        )
    if set(replacements) != EXPECTED_SCATTERS:
        raise ValueError("Expected all six max-pooling sites of the bundled FRNet")

    dimensions = "frnet_voxel_coors_dimensions"
    axis = "frnet_voxel_row_axis"
    generated = {dimensions, axis}
    if generated & producers.keys():
        raise ValueError("Graph already contains generated FRNet tensors")
    graph.initializer.append(helper.make_tensor(axis, onnx.TensorProto.INT64, [], [0]))
    nodes = []
    for node in graph.node:
        if node.name == count_add.name:
            nodes.append(helper.make_node("Shape", ["voxel_coors"], [dimensions]))
            nodes.append(helper.make_node("Gather", [dimensions, axis], list(node.output)))
        else:
            nodes.append(replacements.get(node.name, node))

    needed = {value.name for value in graph.output}
    live = []
    for node in reversed(nodes):
        if needed.intersection(node.output):
            live.append(node)
            needed.update(name for name in node.input if name)
    del graph.node[:]
    graph.node.extend(reversed(live))
    model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    onnx.checker.check_model(model)
    return len(replacements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use an output path separate from the original ONNX")
    model = onnx.load(args.input)
    count = optimize(model)
    onnx.save(model, args.output)
    print(f"Replaced {count} ScatterElements(max) nodes in {args.output}")


if __name__ == "__main__":
    main()

