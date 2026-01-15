# Inspect ONNXLowLevel ValueInfoProto structure
using ONNXLowLevel

model_path = joinpath(@__DIR__, "models", "rnn_model.onnx")
m = ONNXLowLevel.load(model_path)

println("=== Input Info ===")
io = m.graph.input[1]
println("Type: ", typeof(io))
println("Property names: ", propertynames(io))
println()

# Try to access each property
for prop in propertynames(io)
    val = getproperty(io, prop)
    println("  $prop: ", typeof(val), " = ", val)
end

println("\n=== Exploring type field ===")
# The field is #type, access with var"#type"
t = getproperty(io, Symbol("#type"))
println("TypeProto: ", typeof(t))
println("TypeProto propertynames: ", propertynames(t))

println("\n=== Exploring TypeProto.value ===")
val = t.value
println("value type: ", typeof(val))
println("value: ", val)
println("value.name: ", val.name)
println("value.value: ", val.value)

tensor_type = val.value
println("\n=== TensorType ===")
println("tensor_type type: ", typeof(tensor_type))
println("tensor_type propertynames: ", propertynames(tensor_type))

if hasproperty(tensor_type, :shape)
    shape = tensor_type.shape
    println("\nShape: ", shape)
    println("Shape type: ", typeof(shape))
    println("Shape propertynames: ", propertynames(shape))
    
    if hasproperty(shape, :dim)
        println("\nDims:")
        for d in shape.dim
            println("  dim: ", d, " type: ", typeof(d))
            println("  dim propertynames: ", propertynames(d))
            # Try to get the actual value
            if hasproperty(d, :value)
                println("  dim.value: ", d.value)
            end
        end
    end
end
