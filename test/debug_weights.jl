using ONNXLowLevel

model_path = joinpath(@__DIR__, "models", "rnn_model.onnx")
m = ONNXLowLevel.load(model_path)
graph = m.graph

println("=== Inspecting FC Layer ===\n")

# Find initializers (weights)
initializers = Dict{String, Any}()
for init in graph.initializer
    initializers[init.name] = init
    println("Initializer: $(init.name) -> dims=$(init.dims)")
end

println()

# Find Gemm node
for node in graph.node
    if node.op_type == "Gemm"
        println("Node: $(node.name)")
        println("  Op type: $(node.op_type)")
        println("  Inputs: $(node.input)")
        println("  Outputs: $(node.output)")
        
        println("\n  Attributes:")
        for attr in node.attribute
            if attr.name in ["transA", "transB", "alpha", "beta"]
                println("    $(attr.name): $(attr.i)")
            end
        end
    end
end
