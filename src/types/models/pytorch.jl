using PythonCall

struct PyTorchModel <: NNModel
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

function PyTorchModel(model_load_path::String)
    pytorch_model = load_pytorch(model_load_path)
    return pytorch_model
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
        model_data = torch.load(model_path)
        println("Successfully loaded PyTorch model from $model_path")
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