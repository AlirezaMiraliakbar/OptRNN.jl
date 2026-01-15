using ONNXLowLevel

model_path = joinpath(@__DIR__, "models", "rnn_model.onnx")
m = ONNXLowLevel.load(model_path)
graph = m.graph

println("=== Input shape extraction ===")
io = graph.input[1]
type_proto = getproperty(io, Symbol("#type"))
tensor_type = type_proto.value.value

println("Input name: ", io.name)
for (i, dim) in enumerate(tensor_type.shape.dim)
    println("  dim[$i].value = ", dim.value)
    println("  dim[$i].value.name = ", dim.value.name)
    println("  dim[$i].value.value = ", dim.value.value)
end

println("\n=== Output shape extraction ===")
out_io = graph.output[1]
out_type = getproperty(out_io, Symbol("#type"))
out_tensor = out_type.value.value

println("Output name: ", out_io.name)
for (i, dim) in enumerate(out_tensor.shape.dim)
    println("  dim[$i].value = ", dim.value)
    println("  dim[$i].value.name = ", dim.value.name)
    println("  dim[$i].value.value = ", dim.value.value)
end

println("\n=== Testing parse_io_info ===")
using OptRNN: parse_io_info
result = parse_io_info(graph.input)
println("Input parse result: ", result)

out_result = parse_io_info(graph.output)
println("Output parse result: ", out_result)
