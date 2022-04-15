import onnx_graphsurgeon as gs
import numpy as np
import onnx

graph = gs.import_onnx(onnx.load("/workspace/encoder.onnx"))


#print graph

for node in graph.nodes:
    #print(node.name)
    if node.name == "Not_30":
        not30 = node
    if node.name == "Slice_79":
        slice79 = node
    if node.name == "Slice_84":
        slice84 = node
    if node.name == "Cast_194":
        cast194 = node

target_constant = ["Constant_75", "Constant_76", "Constant_77","Constant_78"]
constants = [node for node in graph.nodes if node.name in target_constant]


cast_out = gs.Variable("cast_out", dtype=np.int64)
cast1 = gs.Node(op="Cast", name = "Not30_slice79", attrs={"to": 7}, inputs=not30.outputs, outputs=[cast_out])
graph.nodes.append(cast1)
slice79.inputs.clear()
slice79.inputs = [cast1.outputs[0], constants[1].outputs[0], constants[2].outputs[0], constants[0].outputs[0], constants[3].outputs[0]]


target_nots = ["Not_193", "Not_204", "Not_350", "Not_361", "Not_507", "Not_518", "Not_664", "Not_675", "Not_821","Not_832","Not_978","Not_989","Not_1135","Not_1146","Not_1292","Not_1303","Not_1449","Not_1460","Not_1606","Not_1617","Not_1763","Not_1774","Not_1920","Not_1931"]
nots = [node for node in graph.nodes if node.name in target_nots]
target_casts = ["Cast_194", "Cast_206", "Cast_351", "Cast_363", "Cast_508", "Cast_520", "Cast_665", "Cast_677", "Cast_822","Cast_834","Cast_979","Cast_991","Cast_1136","Cast_1148","Cast_1293","Cast_1305","Cast_1450","Cast_1462","Cast_1607","Cast_1619","Cast_1764","Cast_1776","Cast_1921","Cast_1933"]
casts = [node for node in graph.nodes if node.name in target_casts]

target_wheres = ["Where_196", "Where_208", "Where_353", "Where_365", "Where_510","Where_522", "Where_667", "Where_679", "Where_824", "Where_836", "Where_981", "Where_993", "Where_1138", "Where_1150", "Where_1295", "Where_1307", "Where_1452", "Where_1464", "Where_1609", "Where_1621", "Where_1766", "Where_1778", "Where_1923", "Where_1935"]
wheres = [node for node in graph.nodes if node.name in target_wheres]

for i in range(0, len(nots)):
    not_in = nots[i].inputs
    not_out = nots[i].outputs
    cast_in = casts[i].inputs
    cast_out = casts[i].outputs

    casts[i].inputs = not_in
    nots[i].inputs = cast_out

    #print(wheres[0].i(1).outputs[0])
    #print(wheres[0].i(2).outputs[0])
    out1 = wheres[i].i(1).outputs[0]
    out2 = wheres[i].i(2).outputs[0]
    wheres[i].inputs.clear()

    wheres[i].inputs = [not_out[0], out1, out2]
#print(casts[0])
#print(nots[0])
#print(wheres[0])




print(cast194)
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "/target/encoder_fp16.onnx")
