# any other utility functions for model parsing and analysis can go here
using PythonCall

function detect_actfunc(onnx_model::Py)
    # Define common activation functions in ONNX
    act_library = ["relu", "sigmoid", "tanh", "leakyrelu", "elu", "selu", 
                   "softmax", "softplus", "softsign", "prelu", "gelu"]
    
    # Extract model structure
    graph = onnx_model.graph
    nodes = graph.node
    
    # Store activation functions and their orders
    actfuncs = []
    #TODO: since the graph also gives lstm, it is nice to detect the type of recurrent neural net here so we can formluate accordingly
    for (i, node) in enumerate(nodes)
        node_type = lowercase(pyconvert(String, node.op_type))
        
        if node_type in act_library
            # Get node name, inputs and outputs for reference
            node_name = pyconvert(String, node.name)
            inputs = [pyconvert(String, input) for input in node.input]
            outputs = [pyconvert(String, output) for output in node.output]
            
            # Store information about this activation function
            push!(actfuncs, Dict(
                "index" => i,
                "type" => node_type,
                "name" => node_name,
                "inputs" => inputs,
                "outputs" => outputs
            ))
            
            println("Found activation: $node_type at position $i")
        end
    end
    
    return actfuncs
end