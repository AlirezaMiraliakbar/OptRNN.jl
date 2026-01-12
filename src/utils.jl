
using PythonCall

abstract type NNModel end
abstract type ActFunc end

struct ONNXModel <: NNModel
    input_size::Int64;
    output_size::Int64;
    hidden_sizes::Vector{Int64};
    inputs::Vector{Dict{String, Any}};
    outputs::Vector{Dict{String, Any}};
    operations::Vector{Dict{String, Any}};
    num_inputs::Int64;
    num_outputs::Int64;
end

struct PytorchModel <: NNModel
    input_size::Int64;
    output_size::Int64;
    hidden_sizes::Vector{Int64};
    sequence_length::Int64;
    num_layers::Int64;
    architecture::String;  # "RNN", "LSTM", "GRU"
    inputs::Vector{Dict{String, Any}};
    outputs::Vector{Dict{String, Any}};
    operations::Vector{Dict{String, Any}};
    num_inputs::Int64;
    num_outputs::Int64;
end

function PytorchModel(model_load_path::String)
    pytorch_model = load_pytorch(model_load_path)
    return pytorch_model
end

"""
        load_onnx(path::String) -> ONNXModel

    Load an ONNX model from the specified file path and parse its structure.

    # Arguments
    - `path::String`: The file path to the ONNX model file.

    # Returns
    - `ONNXModel`: A struct containing the parsed ONNX model with the following fields:
      - `input_size::Int64`: Size of the input layer
      - `output_size::Int64`: Size of the output layer
      - `hidden_sizes::Vector{Int64}`: Sizes of hidden layers (extracted from RNN/LSTM/GRU operations)
      - `inputs::Vector{Dict{String, Any}}`: Information about input tensors (names, shapes)
      - `outputs::Vector{Dict{String, Any}}`: Information about output tensors (names, shapes)
      - `operations::Vector{Dict{String, Any}}`: List of operations in the computational graph with weights/biases
      - `num_inputs::Int64`: Number of input tensors
      - `num_outputs::Int64`: Number of output tensors

    # Example


"""
function load_onnx(model_path::String)

    function parse_onnx_structure()
        onnx = pyimport("onnx")
        model = onnx.load(model_path)
        graph = model.graph
        
        # Get input information
        inputs = []
        for input in graph.input
            input_name = pyconvert(String, input.name)
            input_shape = [pyconvert(Int, dim.dim_value) for dim in input.type.tensor_type.shape.dim if pyconvert(Int, dim.dim_value) > 0]
            push!(inputs, Dict("name" => input_name, "shape" => input_shape))
        end
        
        # Get output information
        outputs = []
        for output in graph.output
            output_name = pyconvert(String, output.name)
            output_shape = [pyconvert(Int, dim.dim_value) for dim in output.type.tensor_type.shape.dim if pyconvert(Int, dim.dim_value) > 0]
            push!(outputs, Dict("name" => output_name, "shape" => output_shape))
        end
        
        # Parse operations and extract hidden sizes
        operations = []
        hidden_sizes = []

        n_rnn_modules = 0
        n_module_layer = 0
        rnn_module_info = []

        for node in graph.node

            op_type = pyconvert(String, node.op_type)
            node_name = pyconvert(String, node.name)
            node_inputs = [pyconvert(String, inp) for inp in node.input]
            node_outputs = [pyconvert(String, out) for out in node.output]
            

            push!(operations, Dict(
                "type" => op_type,
                "name" => node_name,
                "inputs" => node_inputs,
                "outputs" => node_outputs
            ))
            # rnn operations are saved as RNN (1st layer), RNN_1 (second layer), RNN_2 (third layer), ...
            

             

            # Extract X from node_name pattern /rnnX/RNN_Y
            if occursin("rnn", lowercase(node_name)) && lowercase(op_type) == "rnn"
                # Match pattern like /rnn0/RNN_1, /rnn1/RNN, /rnn/RNN_0
                rnn_match = match(r"/rnn(\d*)/rnn(?:_(\d+))?", lowercase(node_name))
                if rnn_match !== nothing
                    # Extract X (module number) - default to 0 if not present
                    X = isnothing(rnn_match.captures[1]) || isempty(rnn_match.captures[1]) ? 0 : parse(Int, rnn_match.captures[1])
                    # Extract Y (layer number) - default to 0 if not present
                    Y = isnothing(rnn_match.captures[2]) ? 0 : parse(Int, rnn_match.captures[2])
                    
                    # Update the maximum module number seen
                    n_rnn_modules = max(n_rnn_modules, X == 0 ? X + 1 : X)
                    if X != 0
                        push!(rnn_module_info, X => Y+1)
                    elseif X == 0
                        push!(rnn_module_info, X+1 => Y+1)
                    end
                    println("Found RNN module $X, layer $(Y+1) in node: $node_name")
                else
                    # If pattern doesn't match expected format
                    X = 0
                    Y = 0
                    n_rnn_modules = max(n_rnn_modules, 1)  # At least one RNN module found
                    println("Warning: RNN node name doesn't match expected pattern: $node_name")
                end
                
            end
            #TODO: implement the above logic for lstm and gru

            

            if lowercase(op_type) == "rnn"
                # based on the node_name we can get which rnn we are working with 

                node_weights = Dict()
                node_biases = Dict()
                
                for initializer in graph.initializer
                    init_name = pyconvert(String, initializer.name)
                    index = 0
                    if init_name in node_inputs
                        index += 1
                        # println(init_name)
                        param_array = pyconvert(Array, onnx.numpy_helper.to_array(initializer))
                        # display(param_array)
                        # println(" =================================================== ")
                        if occursin(lowercase(op_type), lowercase(init_name))
                            # we are not sure what indices the ONNX actually have from 3 - 6 possible inputs. One idea that I have is to classify any inputs based on                             
                            if index == 1
                                # W: The weight tensor for input gate. Concatenation of Wi and WBi (if bidirectional). The tensor has shape [num_directions, hidden_size, input_size].
                                node_direction = size(param_array,1)
                                node_hidden_size = size(param_array, 2)
                                node_input_size = size(param_array, 3)

                                if node_direction > 1
                                    # separating out Wi from WBi for easier use later 
                                    Wi = param_array[1,:,:]
                                    WBi = param_array[node_direction, :, :]
                                    node_weights["Wi"] = Wi
                                    node_weights["WBi"] = WBi
                                else
                                    node_weights["Wi"] = param_array[1,:,:];
                                end

                            elseif index == 2
                                # R: the recurrence weight tensor. Concatenation of Ri and RBi (if bidirectional). The tensor has shape [num_directions, hidden_size, hidden_size].
                                init_name = "R"
                                # checking based on previous information extractd from W
                                @assert size(param_array, 1) == node_direction "bidirectional index does not match from W"
                                @assert size(param_array, 2) == node_hidden_size "the hidden sizes does not match from W"
                                @assert size(param_array, 3) == node_hidden_size "hidden size does not match from W"

                                node_weights[init_name] = param_array
                            elseif index == 3
                                # B: The bias tensor for input gate. Concatenation of [Wbi, Rbi] and [WBbi, RBbi] (if bidirectional). The tensor has shape [num_directions, 2*hidden_size]. Optional: If not specified - assumed to be 0.
                                init_name = "B"
                                node_biases[init_name] = param_array
                            elseif index > 6 || index < 3
                                error("RNN should have 3-6 tags for W, R and B. Check the ONNX documentation: https://onnx.ai/onnx/operators/onnx__RNN.html. If you think there is an implementation problem from our side please raise an issue at:")
                            end

                        end
                    end
                end
            elseif lowercase(op_type) == "lstm"


            elseif lowercase(op_type) == "gru"
                # gru code goes here
            end
        end

        # Get distinct first keys (module numbers) from rnn_module_info
        distinct_modules = unique([pair.first for pair in rnn_module_info])
        module_counts = Dict(mod => count(p -> p.first == mod, rnn_module_info) for mod in distinct_modules)

        
        n_modules = length(distinct_modules)

        # Check if any RNN-type operations are present
        rnn_operations = ["lstm", "rnn", "gru"]
        has_rnn = any(op -> lowercase(op["type"]) in rnn_operations, operations)

        if !has_rnn
            error("The given ONNX model does not have any recurrent-based layer (e.g. LSTM). OptRNN requires explicit RNN node types to function. When you are exporting ONNX, use the legacy mode as True by setting dynamo=False in PyTorch. There are numerous other alternatives that you can use. check OMLT project: https://github.com/cog-imperial/OMLT")
        end

        structure_info = Dict(
            "inputs" => inputs,
            "outputs" => outputs,
            "n_modules" => n_modules,
            "layer_per_module" => module_counts,
            "operations" => operations,
            "input_size" => input_size,
            "output_size" => output_size,
            "hidden_sizes" => hidden_sizes,
            "num_inputs" => length(inputs),
            "num_outputs" => length(outputs)
        )
        
        return operations
    end

    # # Call the parsing function and create ONNXModel
    operations = parse_onnx_structure()
    # # For now, using a simple DAG placeholder - you'll need to implement proper DAG structure
    # return ONNXModel(
    #     structure["input_size"],
    #     structure["output_size"], 
    #     structure["hidden_sizes"],
    #     structure["inputs"],
    #     structure["outputs"],
    #     structure["operations"],
    #     structure["num_inputs"],
    #     structure["num_outputs"]
    # )
    return operations
end

function load_nn_model(path::String)
    onnx = pyimport("onnx")
    nn_model = onnx.load(model_load_path)
    act_funcs = detect_actfunc(nn_model)
    # Extract model structure
    graph = nn_model.graph

    # Extract weights and biases
    weights = Dict()
    biases = Dict()
    lstm_params = Dict()
    global linear_counter = 0
    global bias_index = 0
    # linking keys with indices for FC layers
    weight_index_dict = Dict()
    biases_index_dict = Dict()
    for initializer in graph.initializer
        name = pyconvert(String,initializer.name)
        
        param_array = pyconvert(Array, onnx.numpy_helper.to_array(initializer))
        # Convert ONNX tensor to Julia array
        if occursin("weight", lowercase(name))
            println(name)
            @show typeof(name)
            weights[name] = pyconvert(Array, onnx.numpy_helper.to_array(initializer))
            global linear_counter += 1
            weight_index_dict[linear_counter] = name
        elseif occursin("bias", lowercase(name))
            biases[name] = pyconvert(Array, onnx.numpy_helper.to_array(initializer))
            global bias_index += 1
            biases_index_dict[bias_index] = name

        elseif occursin("lstm", lowercase(name)) || length(size(param_array)) >= 2
            # LSTM parameters are often multi-dimensional or have "lstm" in name
            lstm_params[name] = param_array
            println("  -> Classified as: LSTM parameter")
        else
            error("More support is needed! please raise an issue on github: ")
        end
    end

    #TODO: need to dynamically get the key here later
    W_is = lstm_params["onnx::LSTM_111"]

    batch_size = size(W_is, 1)
    hidden_size = Int64(size(W_is, 2) / 4)
    input_size = size(W_is, 3)

    W_ii = W_is[1,1:hidden_size,:]
    W_if = W_is[1,(2*hidden_size+1):(3*hidden_size),:]
    W_ig = W_is[1,(3*hidden_size+1):(4*hidden_size),:]
    W_io = W_is[1,hidden_size+1:(2*hidden_size),:]

    W_hs = lstm_params["onnx::LSTM_112"]

    W_hi = W_hs[1,1:hidden_size,:]
    W_hf = W_hs[1,(2*hidden_size+1):(3*hidden_size),:]
    W_hg = W_hs[1,(3*hidden_size+1):(4*hidden_size),:]
    W_ho = W_hs[1,hidden_size+1:(2*hidden_size),:]

    bs = lstm_params["onnx::LSTM_113"]

    b_ii = bs[1:hidden_size]
    b_if = bs[(2*hidden_size+1):(3*hidden_size)]
    b_ig = bs[(3*hidden_size+1):(4*hidden_size)]
    b_io = bs[hidden_size+1:(2*hidden_size)]

    b_hi = bs[(4*hidden_size+1):(5*hidden_size)]
    b_ho = bs[(5*hidden_size+1):(6*hidden_size)]
    b_hg = bs[(6*hidden_size+1):(7*hidden_size)]
    b_ho = bs[(7*hidden_size+1):(8*hidden_size)]

    # extracting other model components 
    println("Number of linear layers = $linear_counter")    

    a = biases_index_dict[linear_counter]
    output_size = size(biases[a],1)
    println("output_size = $output_size")
    param_dict = Dict(
        "input_size" => input_size,
        "hidden_size" => hidden_size,
        "output_size" => output_size,
        "n_linear_layer" => linear_counter,
        "w_ii" => W_ii,
        "w_if" => W_if,
        "w_ig" => W_ig,
        "w_io" => W_io,
        "w_hi" => W_hi,
        "w_hf" => W_hf,
        "w_hg" => W_hg,
        "w_ho" => W_ho,
        "b_ii" => b_ii,
        "b_if" => b_if,
        "b_ig" => b_ig,
        "b_io" => b_io,
        "b_hi" => b_hi,
        "b_hf" => b_hf,
        "b_hg" => b_hg,
        "b_ho" => b_ho,
        "fc_weights" => weights,
        "fc_biases" => biases
    )
    return param_dict
end


"""
    load_pytorch(model_path::String; sequence_length::Union{Int64, Nothing}=nothing, device::String="cpu") -> PytorchModel

Load a PyTorch model from the specified file path and extract properties needed for optimization formulation.

# Arguments
- `model_path::String`: The file path to the PyTorch model file (.pth, .pt, or .pkl)
- `sequence_length::Union{Int64, Nothing}`: Optional sequence length. If not provided, will attempt to infer from model structure (default: nothing)
- `device::String`: Device to load the model on ("cpu" or "cuda") (default: "cpu")

# Returns
- `PytorchModel`: A struct containing the parsed PyTorch model with the following fields:
  - `input_size::Int64`: Size of the input layer
  - `output_size::Int64`: Size of the output layer
  - `hidden_sizes::Vector{Int64}`: Sizes of hidden layers
  - `sequence_length::Int64`: Sequence length for the RNN
  - `num_layers::Int64`: Number of recurrent layers
  - `architecture::String`: Model architecture type ("RNN", "LSTM", or "GRU")
  - `inputs::Vector{Dict{String, Any}}`: Information about input tensors
  - `outputs::Vector{Dict{String, Any}}`: Information about output tensors
  - `operations::Vector{Dict{String, Any}}`: List of operations in the model
  - `num_inputs::Int64`: Number of input tensors
  - `num_outputs::Int64`: Number of output tensors

# Examples
```julia
model = load_pytorch("model.pth")
model = load_pytorch("model.pth"; sequence_length=100)
```
"""
function load_pytorch(model_path::String; sequence_length::Union{Int64, Nothing}=nothing, device::String="cpu")
    torch = pyimport("torch")
    
    # Load the model with weights_only=false for PyTorch 2.6+ compatibility
    try
        model_data = torch.load(model_path; map_location=device, weights_only=false)
    catch e
        error("Failed to load PyTorch model from $model_path: $e")
    end

    println(model_data)

    # Initialize containers
    inputs = []
    outputs = []
    operations = []
    hidden_sizes = Int64[]
    
    # Determine if model_data is a state_dict or a full model
    state_dict = nothing
    model_arch = nothing
    
    # Try to check if it's a model object with state_dict method
    try
        if pyhasattr(model_data, "state_dict")
            # Full model object
            println("Loaded full PyTorch model object")
            model_arch = model_data
            state_dict = pyconvert(Dict, model_data.state_dict())
        else
            # Assume it's a state_dict
            println("Assuming loaded data is state_dict")
            state_dict = pyconvert(Dict, model_data)
        end
    catch
        # If conversion fails, try as state_dict directly
        try
            state_dict = pyconvert(Dict, model_data)
        catch e
            error("Unsupported PyTorch model format. Expected state_dict (Dict) or model object. Error: $e")
        end
    end
    
    # Convert state_dict keys to Julia strings for easier processing
    state_dict_julia = Dict{String, Any}()
    for (k, v) in state_dict
        key_str = pyconvert(String, string(k))
        try
            # Try to convert tensor to Julia array
            if pyhasattr(v, "numpy")
                state_dict_julia[key_str] = pyconvert(Array, v.numpy())
            else
                state_dict_julia[key_str] = v
            end
        catch
            state_dict_julia[key_str] = v
        end
    end
    
    # Detect architecture type and extract properties
    architecture = "Unknown"
    num_layers = 0
    input_size = 0
    output_size = 0
    
    # Look for RNN/LSTM/GRU layers in state_dict keys
    rnn_keys = [k for k in keys(state_dict_julia) if occursin(r"rnn|lstm|gru", lowercase(k))]
    
    if isempty(rnn_keys)
        error("No RNN, LSTM, or GRU layers found in the model. OptRNN only supports recurrent neural networks.")
    end
    
    # Determine architecture from first RNN key
    first_rnn_key = rnn_keys[1]
    if occursin("lstm", lowercase(first_rnn_key))
        architecture = "LSTM"
    elseif occursin("gru", lowercase(first_rnn_key))
        architecture = "GRU"
    elseif occursin("rnn", lowercase(first_rnn_key))
        architecture = "RNN"
    end
    
    # Extract layer information from state_dict
    # PyTorch RNN layers typically have keys like:
    # - weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0 (layer 0)
    # - weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1 (layer 1)
    # etc.
    
    layer_patterns = [
        r"weight_ih_l(\d+)",  # Input-to-hidden weights
        r"weight_hh_l(\d+)",  # Hidden-to-hidden weights
        r"bias_ih_l(\d+)",    # Input-to-hidden bias
        r"bias_hh_l(\d+)"     # Hidden-to-hidden bias
    ]
    
    layer_numbers = Int64[]
    for key in keys(state_dict_julia)
        for pattern in layer_patterns
            m = match(pattern, key)
            if m !== nothing
                layer_num = parse(Int, m.captures[1])
                push!(layer_numbers, layer_num)
            end
        end
    end
    
    if !isempty(layer_numbers)
        num_layers = maximum(layer_numbers) + 1  # Layers are 0-indexed
    else
        # Fallback: count unique layer identifiers
        num_layers = 1
    end
    
    # Extract hidden sizes and input sizes from weight matrices
    # weight_ih_l0 shape: [4*hidden_size, input_size] for LSTM, [hidden_size, input_size] for RNN/GRU
    # weight_hh_l0 shape: [4*hidden_size, hidden_size] for LSTM, [hidden_size, hidden_size] for RNN/GRU
    
    weight_ih_keys = [k for k in keys(state_dict_julia) if occursin("weight_ih_l0", k)]
    weight_hh_keys = [k for k in keys(state_dict_julia) if occursin("weight_hh_l0", k)]
    
    hidden_size = 0  # Initialize to avoid undefined variable
    
    if !isempty(weight_ih_keys)
        weight_ih_key = weight_ih_keys[1]
        if haskey(state_dict_julia, weight_ih_key) && isa(state_dict_julia[weight_ih_key], Array)
            weight_ih = state_dict_julia[weight_ih_key]
            weight_ih_shape = size(weight_ih)
            
            if architecture == "LSTM"
                # LSTM: weight_ih shape is [4*hidden_size, input_size]
                hidden_size = Int(weight_ih_shape[1] ÷ 4)
            else
                # RNN/GRU: weight_ih shape is [hidden_size, input_size]
                hidden_size = Int(weight_ih_shape[1])
            end
            
            input_size = Int(weight_ih_shape[2])
            
            # Extract hidden sizes for all layers
            for layer_idx in 0:(num_layers-1)
                weight_ih_key_layer = replace(weight_ih_key, "l0" => "l$layer_idx")
                if haskey(state_dict_julia, weight_ih_key_layer) && isa(state_dict_julia[weight_ih_key_layer], Array)
                    weight_ih_layer = state_dict_julia[weight_ih_key_layer]
                    if architecture == "LSTM"
                        hidden_size_layer = Int(size(weight_ih_layer, 1) ÷ 4)
                    else
                        hidden_size_layer = Int(size(weight_ih_layer, 1))
                    end
                    push!(hidden_sizes, hidden_size_layer)
                else
                    # Assume same hidden size for all layers
                    push!(hidden_sizes, hidden_size)
                end
            end
        else
            error("Failed to extract weight_ih_l0 from PyTorch model state_dict. Model structure may be unsupported.")
        end
    else
        error("No weight_ih_l0 keys found in PyTorch model state_dict. Model may not be an RNN/LSTM/GRU model or uses non-standard naming.")
    end
    
    # Extract output size (try to find final linear layer)
    linear_keys = [k for k in keys(state_dict_julia) if occursin("weight", lowercase(k)) && 
                   (occursin("linear", lowercase(k)) || occursin("fc", lowercase(k)) || occursin("out", lowercase(k)))]
    
    if !isempty(linear_keys)
        # Use the last linear layer's output size
        last_linear_key = linear_keys[end]
        if haskey(state_dict_julia, last_linear_key) && isa(state_dict_julia[last_linear_key], Array)
            linear_weight = state_dict_julia[last_linear_key]
            output_size = Int(size(linear_weight, 1))
        end
    else
        # If no linear layer found, output size equals last hidden size
        output_size = isempty(hidden_sizes) ? hidden_size : hidden_sizes[end]
    end
    
    # Infer sequence_length if not provided
    seq_length = sequence_length
    if seq_length === nothing
        # Try to infer from model structure or use a default
        # This is tricky without model architecture info, so we'll use a placeholder
        seq_length = 0  # Will need to be set by user or inferred from input
    end
    
    # Create input/output info
    push!(inputs, Dict("name" => "input", "shape" => [seq_length, input_size]))
    push!(outputs, Dict("name" => "output", "shape" => [seq_length, output_size]))
    
    # Create operations list from state_dict keys
    for (key, value) in state_dict_julia
        if isa(value, Array)
            push!(operations, Dict(
                "type" => "parameter",
                "name" => key,
                "shape" => collect(size(value))
            ))
        end
    end
    
    return PytorchModel(
        input_size,
        output_size,
        hidden_sizes,
        seq_length,
        num_layers,
        architecture,
        inputs,
        outputs,
        operations,
        length(inputs),
        length(outputs)
    )
end

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