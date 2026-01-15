"""
    ONNX Graph Parser for RNN Models (using ONNXLowLevel)
    
    This module provides functionality to parse ONNX computational graphs,
    identify RNN layers (RNN, LSTM, GRU), extract weights/biases, and 
    prepare the information needed for JuMP constraint generation.
    
    Only tracks actual mathematical operations, ignoring infrastructure ops
    like Transpose, Reshape, Squeeze, Gather, Concat, etc.
"""

using ONNXLowLevel
using LinearAlgebra

# ============================================================================
# Constants - Mathematical vs Infrastructure Operations
# ============================================================================

"""
Operations that perform actual mathematical transformations.
These are the only operations we track for JuMP constraint generation.
"""
const MATHEMATICAL_OPS = Set([
    # RNN operations
    "RNN", "LSTM", "GRU",
    # Linear algebra
    "Gemm", "MatMul", "MatMulInteger",
    # Element-wise math
    "Add", "Sub", "Mul", "Div",
    # Activations
    "Relu", "Sigmoid", "Tanh", "Softmax", "LeakyRelu", "Elu", "Selu",
    "HardSigmoid", "Softsign", "Softplus",
    # Reductions
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
    # Normalization
    "BatchNormalization", "LayerNormalization",
    # Convolutions (if present)
    "Conv", "ConvTranspose"
])

"""
Infrastructure operations that don't affect mathematical semantics.
These are skipped during parsing.
"""
const INFRASTRUCTURE_OPS = Set([
    "Transpose", "Reshape", "Squeeze", "Unsqueeze",
    "Shape", "Gather", "Concat", "Split", "Slice",
    "Constant", "Identity", "Flatten", "Expand",
    "Cast", "Tile", "Pad", "ConstantOfShape"
])

# ============================================================================
# Data Structures
# ============================================================================

"""
    RNNLayerInfo

Stores information about a single RNN/LSTM/GRU layer extracted from ONNX.
"""
struct RNNLayerInfo
    name::String
    op_type::Symbol  # :RNN, :LSTM, :GRU
    layer_index::Int  # 0-indexed layer within module
    module_index::Int  # Which RNN module (for stacked RNNs)
    
    # Dimensions
    input_size::Int
    hidden_size::Int
    num_directions::Int  # 1 for unidirectional, 2 for bidirectional
    
    # Weights (Julia column-major order, reversed from ONNX row-major)
    # ONNX: [num_directions, hidden_size*, input_size]
    # Julia: [input_size, hidden_size*, num_directions]
    W::Array{Float32}  # Input weights
    R::Array{Float32}  # Recurrent weights [hidden_size, hidden_size*, num_directions]
    B::Union{Array{Float32}, Nothing}  # Biases (optional)
    
    # For LSTM/GRU: initial states (optional)
    initial_h::Union{Array{Float32}, Nothing}
    initial_c::Union{Array{Float32}, Nothing}  # Only for LSTM
    
    # Connection info (only mathematical predecessors/successors)
    input_names::Vector{String}
    output_names::Vector{String}
    
    # Activation functions
    activations::Vector{String}
end

"""
    FCLayerInfo

Stores information about a fully connected layer (Gemm/MatMul operation).
"""
struct FCLayerInfo
    name::String
    op_type::Symbol  # :Gemm or :MatMul
    input_size::Int
    output_size::Int
    W::Array{Float32}
    b::Union{Vector{Float32}, Nothing}
    input_names::Vector{String}
    output_names::Vector{String}
end

"""
    ActivationInfo

Stores information about standalone activation operations.
"""
struct ActivationInfo
    name::String
    op_type::Symbol  # :Relu, :Sigmoid, :Tanh, etc.
    input_names::Vector{String}
    output_names::Vector{String}
end

"""
    ParsedONNXModel

Complete parsed representation of an ONNX model containing RNN layers.
Only includes mathematical operations.
"""
struct ParsedONNXModel
    # Layer information (only mathematical ops)
    rnn_layers::Vector{RNNLayerInfo}
    fc_layers::Vector{FCLayerInfo}
    activations::Vector{ActivationInfo}
    
    # Model structure
    num_rnn_modules::Int
    layers_per_module::Dict{Int, Int}
    
    # Global dimensions
    model_input_size::Int
    model_output_size::Int
    sequence_length::Union{Int, Nothing}  # May be dynamic
    
    # Graph connectivity (only mathematical ops)
    input_info::Vector{Dict{String, Any}}
    output_info::Vector{Dict{String, Any}}
    
    # Computation order (only mathematical layer names)
    math_operation_order::Vector{String}
    
    # Raw ONNX graph for advanced queries
    raw_graph::ONNXLowLevel.GraphProto
end

# ============================================================================
# Core Parsing Functions
# ============================================================================

"""
    is_mathematical_op(op_type::String) -> Bool

Check if an operation is a mathematical operation worth tracking.
"""
is_mathematical_op(op_type::String) = op_type in MATHEMATICAL_OPS

"""
    parse_onnx_graph(model_path::String) -> ParsedONNXModel

Parse an ONNX model and extract all RNN layer information needed for JuMP constraints.
Only tracks mathematical operations (RNN, LSTM, GRU, Gemm, activations, etc.).

# Arguments
- `model_path::String`: Path to the ONNX model file

# Returns
- `ParsedONNXModel`: Complete parsed model with layer info, weights, and connectivity
"""
function parse_onnx_graph(model_path::String)
    # Load the ONNX model using ONNXLowLevel
    model = ONNXLowLevel.load(model_path)
    graph = model.graph
    
    # Build initializer lookup for weights/biases
    initializers = build_initializer_dict(graph)
    
    # Parse only mathematical operations
    rnn_layers = RNNLayerInfo[]
    fc_layers = FCLayerInfo[]
    activations = ActivationInfo[]
    math_operation_order = String[]
    
    # Track RNN module/layer structure
    rnn_module_info = Dict{Int, Vector{Int}}()
    
    for node in graph.node
        op_type = node.op_type
        node_name = node.name
        
        # Skip non-mathematical operations
        if !is_mathematical_op(op_type)
            continue
        end
        
        push!(math_operation_order, node_name)
        
        if op_type in ["RNN", "LSTM", "GRU"]
            layer_info = parse_rnn_node(node, initializers, graph)
            push!(rnn_layers, layer_info)
            
            # Track module structure
            mod_idx = layer_info.module_index
            if !haskey(rnn_module_info, mod_idx)
                rnn_module_info[mod_idx] = Int[]
            end
            push!(rnn_module_info[mod_idx], layer_info.layer_index)
            
        elseif op_type in ["Gemm", "MatMul"]
            fc_info = parse_fc_node(node, initializers, op_type)
            if !isnothing(fc_info)
                push!(fc_layers, fc_info)
            end
            
        elseif op_type in ["Relu", "Sigmoid", "Tanh", "Softmax", "LeakyRelu", 
                           "Elu", "Selu", "HardSigmoid", "Softsign", "Softplus"]
            act_info = parse_activation_node(node)
            push!(activations, act_info)
        end
    end
    
    # Calculate summary statistics
    num_rnn_modules = isempty(rnn_module_info) ? 0 : length(rnn_module_info)
    layers_per_module = Dict(k => length(v) for (k, v) in rnn_module_info)
    
    # Extract input/output info
    input_info = parse_io_info(graph.input)
    output_info = parse_io_info(graph.output)
    
    # Determine model dimensions
    model_input_size = get_model_input_size(input_info, rnn_layers)
    model_output_size = get_model_output_size(output_info, fc_layers, rnn_layers)
    sequence_length = get_sequence_length(input_info)
    
    return ParsedONNXModel(
        rnn_layers,
        fc_layers,
        activations,
        num_rnn_modules,
        layers_per_module,
        model_input_size,
        model_output_size,
        sequence_length,
        input_info,
        output_info,
        math_operation_order,
        graph
    )
end

"""
    build_initializer_dict(graph) -> Dict{String, Array}

Create a lookup dictionary from initializer names to their numerical values.
"""
function build_initializer_dict(graph::ONNXLowLevel.GraphProto)
    initializers = Dict{String, Array}()
    
    for init in graph.initializer
        name = init.name
        data = decode_tensor(init)
        if !isnothing(data)
            initializers[name] = data
        end
    end
    
    return initializers
end

"""
    decode_tensor(tensor::ONNXLowLevel.TensorProto) -> Array

Decode an ONNX TensorProto to a Julia array.
"""
function decode_tensor(tensor::ONNXLowLevel.TensorProto)
    dims = Int.(tensor.dims)
    data_type = tensor.data_type
    
    # ONNX data types: 1=FLOAT, 2=UINT8, 3=INT8, 4=UINT16, 5=INT16,
    #                  6=INT32, 7=INT64, 10=FLOAT16, 11=DOUBLE
    
    # Check for raw_data first
    if !isempty(tensor.raw_data)
        return decode_raw_data(tensor.raw_data, data_type, dims)
    end
    
    # Check typed arrays
    if data_type == 1 && !isempty(tensor.float_data)
        return isempty(dims) ? Float32(tensor.float_data[1]) : 
               reshape(Float32.(tensor.float_data), reverse(dims)...)
    elseif data_type == 11 && !isempty(tensor.double_data)
        return isempty(dims) ? Float64(tensor.double_data[1]) :
               reshape(Float64.(tensor.double_data), reverse(dims)...)
    elseif data_type == 7 && !isempty(tensor.int64_data)
        return isempty(dims) ? Int64(tensor.int64_data[1]) :
               reshape(Int64.(tensor.int64_data), reverse(dims)...)
    elseif data_type == 6 && !isempty(tensor.int32_data)
        return isempty(dims) ? Int32(tensor.int32_data[1]) :
               reshape(Int32.(tensor.int32_data), reverse(dims)...)
    end
    
    return nothing
end

"""
    decode_raw_data(raw_data, data_type, dims) -> Array

Decode raw binary data from ONNX tensor based on data type.
"""
function decode_raw_data(raw_data::Vector{UInt8}, data_type::Int32, dims::Vector{Int})
    julia_type = if data_type == 1
        Float32
    elseif data_type == 11
        Float64
    elseif data_type == 6
        Int32
    elseif data_type == 7
        Int64
    elseif data_type == 2
        UInt8
    elseif data_type == 3
        Int8
    else
        @warn "Unsupported ONNX data type: $data_type, treating as Float32"
        Float32
    end
    
    # Reinterpret bytes as the target type
    data = reinterpret(julia_type, raw_data)
    
    # Reshape to ONNX dimensions (ONNX is row-major, Julia is column-major)
    if isempty(dims)
        return copy(data)[1]  # Scalar
    else
        return reshape(copy(data), reverse(dims)...)
    end
end

# ============================================================================
# RNN Node Parsing
# ============================================================================

"""
    parse_rnn_node(node, initializers, graph) -> RNNLayerInfo

Parse a single RNN/LSTM/GRU node and extract all parameters.
"""
function parse_rnn_node(node::ONNXLowLevel.NodeProto, 
                        initializers::Dict{String, Array},
                        graph::ONNXLowLevel.GraphProto)
    node_name = node.name
    op_type = node.op_type
    input_names = collect(node.input)
    output_names = collect(node.output)
    
    # Extract module and layer indices from name
    module_idx, layer_idx = extract_rnn_indices(node_name)
    
    # ONNX RNN input order: X, W, R, B (optional), sequence_lens (optional), initial_h (optional)
    # For LSTM: X, W, R, B, sequence_lens, initial_h, initial_c, P (optional)
    
    # Get W (input weights) - index 1
    W = get_weight_by_index(input_names, initializers, 2)
    if isnothing(W)
        error("Could not find input weight W for RNN node: $node_name")
    end
    
    # Get R (recurrent weights) - index 2
    R = get_weight_by_index(input_names, initializers, 3)
    if isnothing(R)
        error("Could not find recurrent weight R for RNN node: $node_name")
    end
    
    # Get B (biases) - index 3, optional
    B = get_weight_by_index(input_names, initializers, 4)
    
    # Get initial_h - index 5 (0-indexed: 5), optional
    initial_h = get_weight_by_index(input_names, initializers, 6)
    
    # Get initial_c for LSTM - index 6 (0-indexed: 6), optional
    initial_c = nothing
    if op_type == "LSTM"
        initial_c = get_weight_by_index(input_names, initializers, 7)
    end
    
    # Extract dimensions from W
    # ONNX W shape: [num_directions, hidden_size_multiplier * hidden_size, input_size]
    # Julia (after reverse): [input_size, hidden_size_multiplier * hidden_size, num_directions]
    input_size = size(W, 1)
    num_directions = size(W, 3)
    
    # hidden_size calculation depends on op_type
    hidden_multiplier = get_hidden_multiplier(op_type)
    hidden_size = div(size(W, 2), hidden_multiplier)
    
    # Extract activation functions from attributes
    activations = extract_activations_from_attrs(node, op_type)
    
    return RNNLayerInfo(
        node_name,
        Symbol(op_type),
        layer_idx,
        module_idx,
        input_size,
        hidden_size,
        num_directions,
        Float32.(W),
        Float32.(R),
        isnothing(B) ? nothing : Float32.(B),
        isnothing(initial_h) ? nothing : Float32.(initial_h),
        isnothing(initial_c) ? nothing : Float32.(initial_c),
        input_names,
        output_names,
        activations
    )
end

"""
    get_hidden_multiplier(op_type) -> Int

Get the hidden size multiplier for weight matrices based on RNN type.
"""
function get_hidden_multiplier(op_type::String)
    if op_type == "RNN"
        return 1
    elseif op_type == "LSTM"
        return 4  # 4 gates: input, forget, cell, output
    elseif op_type == "GRU"
        return 3  # 3 gates: reset, update, hidden
    else
        error("Unknown RNN type: $op_type")
    end
end

"""
    extract_rnn_indices(node_name) -> (module_idx, layer_idx)

Extract module and layer indices from RNN node name.
Handles patterns like: /rnn/RNN, /rnn/RNN_1, /rnn0/RNN_2, /rnn1/LSTM
"""
function extract_rnn_indices(node_name::String)
    name_lower = lowercase(node_name)
    
    # Pattern 1: /rnn(\d*)/\w+_(\d+) - e.g., /rnn0/RNN_1
    m = match(r"/rnn(\d*)/\w+_(\d+)", name_lower)
    if m !== nothing
        module_idx = isempty(m.captures[1]) ? 1 : parse(Int, m.captures[1]) + 1
        layer_idx = parse(Int, m.captures[2])
        return (module_idx, layer_idx)
    end
    
    # Pattern 2: /rnn(\d*)/\w+ - e.g., /rnn/RNN or /rnn0/LSTM
    m = match(r"/rnn(\d*)/\w+$", name_lower)
    if m !== nothing
        module_idx = isempty(m.captures[1]) ? 1 : parse(Int, m.captures[1]) + 1
        layer_idx = 0  # First layer
        return (module_idx, layer_idx)
    end
    
    # Fallback: look for trailing number
    m = match(r"_(\d+)$", node_name)
    if m !== nothing
        return (1, parse(Int, m.captures[1]))
    end
    
    # Default
    return (1, 0)
end

"""
    get_weight_by_index(input_names, initializers, idx) -> Union{Array, Nothing}

Get weight tensor from initializers using input name at given 1-based index.
"""
function get_weight_by_index(input_names::Vector{String}, initializers::Dict{String, Array}, idx::Int)
    if idx > length(input_names)
        return nothing
    end
    
    name = input_names[idx]
    if isempty(name) || !haskey(initializers, name)
        return nothing
    end
    
    return initializers[name]
end

"""
    extract_activations_from_attrs(node, op_type) -> Vector{String}

Extract activation function names from node attributes.
"""
function extract_activations_from_attrs(node::ONNXLowLevel.NodeProto, op_type::String)
    default_activations = if op_type == "RNN"
        ["Tanh"]
    elseif op_type == "LSTM"
        ["Sigmoid", "Tanh", "Tanh"]  # f, g, h activations
    elseif op_type == "GRU"
        ["Sigmoid", "Tanh"]  # f, g activations
    else
        String[]
    end
    
    # Try to extract from attributes
    for attr in node.attribute
        if attr.name == "activations"
            return String[String(s) for s in attr.strings]
        end
    end
    
    return default_activations
end

# ============================================================================
# FC Layer Parsing (Gemm / MatMul)
# ============================================================================

"""
    parse_fc_node(node, initializers, op_type) -> Union{FCLayerInfo, Nothing}

Parse a Gemm or MatMul node (fully connected layer).
"""
function parse_fc_node(node::ONNXLowLevel.NodeProto, 
                       initializers::Dict{String, Array},
                       op_type::String)
    node_name = node.name
    input_names = collect(node.input)
    output_names = collect(node.output)
    
    W = nothing
    b = nothing
    
    # Find weight and bias in inputs
    for inp_name in input_names
        if haskey(initializers, inp_name)
            data = initializers[inp_name]
            if ndims(data) == 2 && isnothing(W)
                W = Float32.(data)
            elseif ndims(data) == 1 && isnothing(b)
                b = Float32.(vec(data))
            end
        end
    end
    
    if isnothing(W)
        # This might be a MatMul with dynamic inputs (not a learned layer)
        return nothing
    end
    
    # Determine dimensions
    # Julia W shape is reversed from ONNX: if ONNX is [out, in], Julia is [in, out]
    # For Gemm: Y = alpha * A * B^T + beta * C (when transB=1)
    # Or: Y = alpha * A * B + beta * C (when transB=0)
    
    trans_b = false
    if op_type == "Gemm"
        for attr in node.attribute
            if attr.name == "transB"
                trans_b = attr.i == 1
            end
        end
    end
    
    # W in Julia has reversed dims from ONNX
    # If ONNX W was [output_size, input_size] and transB=1:
    #   Julia W is [input_size, output_size] after reverse
    #   The transpose happens in the Gemm operation, so effective shape is correct
    # If ONNX W was [output_size, input_size] and transB=0:
    #   This would be unusual but: Julia W is [input_size, output_size]
    
    # After reverse(dims): size(W) gives us the Julia shape
    # ONNX typical with transB=1: ONNX [out, in] -> Julia [in, out]
    if trans_b
        input_size, output_size = size(W)
    else
        output_size, input_size = size(W)
    end
    
    return FCLayerInfo(
        node_name,
        Symbol(op_type),
        input_size,
        output_size,
        W,
        b,
        input_names,
        output_names
    )
end

# ============================================================================
# Activation Node Parsing
# ============================================================================

"""
    parse_activation_node(node) -> ActivationInfo

Parse a standalone activation node.
"""
function parse_activation_node(node::ONNXLowLevel.NodeProto)
    return ActivationInfo(
        node.name,
        Symbol(node.op_type),
        collect(node.input),
        collect(node.output)
    )
end

# ============================================================================
# Input/Output Parsing
# ============================================================================

"""
    parse_io_info(io_list) -> Vector{Dict{String, Any}}

Parse input or output tensor information.
"""
function parse_io_info(io_list)
    info = Dict{String, Any}[]
    
    for io in io_list
        name = io.name
        shape = Int[]
        
        # Extract shape from type info
        # ONNXLowLevel uses Symbol("#type") for the type field
        type_proto = getproperty(io, Symbol("#type"))
        
        # type_proto.value is a ProtoBuf.OneOf containing tensor_type
        if !isnothing(type_proto) && !isnothing(type_proto.value)
            oneof_value = type_proto.value
            # Check if it's a tensor type (oneof_value.name == :tensor_type)
            if oneof_value.name == :tensor_type
                tensor_type = oneof_value.value
                if hasproperty(tensor_type, :shape) && !isnothing(tensor_type.shape)
                    for dim in tensor_type.shape.dim
                        # dim.value is a ProtoBuf.OneOf with :dim_value or :dim_param
                        if !isnothing(dim.value) && dim.value.name == :dim_value
                            dim_val = dim.value.value
                            if dim_val > 0
                                push!(shape, Int(dim_val))
                            else
                                push!(shape, -1)  # Zero or negative means dynamic
                            end
                        else
                            push!(shape, -1)  # Dynamic dimension (symbolic name)
                        end
                    end
                end
            end
        end
        
        push!(info, Dict("name" => name, "shape" => shape))
    end
    
    return info
end

"""
    get_model_input_size(input_info, rnn_layers) -> Int

Determine the input size of the model.
"""
function get_model_input_size(input_info, rnn_layers)
    # First try from RNN layers
    if !isempty(rnn_layers)
        return rnn_layers[1].input_size
    end
    
    # Try from input info
    if !isempty(input_info)
        shape = input_info[1]["shape"]
        if length(shape) >= 2
            return shape[end]  # Last dimension is typically feature size
        end
    end
    
    return 0
end

"""
    get_model_output_size(output_info, fc_layers, rnn_layers) -> Int

Determine the output size of the model.
"""
function get_model_output_size(output_info, fc_layers, rnn_layers)
    # First try from FC layers (last one)
    if !isempty(fc_layers)
        return fc_layers[end].output_size
    end
    
    # Try from RNN layers (last one)
    if !isempty(rnn_layers)
        return rnn_layers[end].hidden_size
    end
    
    # Try from output info
    if !isempty(output_info)
        shape = output_info[1]["shape"]
        if !isempty(shape)
            return shape[end]
        end
    end
    
    return 0
end

"""
    get_sequence_length(input_info) -> Union{Int, Nothing}

Try to determine sequence length from input shape.
"""
function get_sequence_length(input_info)
    if !isempty(input_info)
        shape = input_info[1]["shape"]
        # Typical RNN input: [batch, seq_len, features] or [seq_len, batch, features]
        if length(shape) >= 2
            seq_dim = shape[2]  # Assuming batch-first
            if seq_dim > 0
                return seq_dim
            end
        end
    end
    return nothing
end

# ============================================================================
# Weight Extraction Utilities for JuMP
# ============================================================================

"""
    extract_rnn_weights(layer::RNNLayerInfo) -> Dict

Extract weights in a format suitable for JuMP constraint generation.
Returns separate weight matrices for each gate (for LSTM/GRU).
"""
function extract_rnn_weights(layer::RNNLayerInfo)
    weights = Dict{String, Any}()
    
    if layer.op_type == :RNN
        weights = extract_vanilla_rnn_weights(layer)
    elseif layer.op_type == :LSTM
        weights = extract_lstm_weights(layer)
    elseif layer.op_type == :GRU
        weights = extract_gru_weights(layer)
    end
    
    weights["hidden_size"] = layer.hidden_size
    weights["input_size"] = layer.input_size
    weights["num_directions"] = layer.num_directions
    weights["activations"] = layer.activations
    
    return weights
end

"""
    extract_vanilla_rnn_weights(layer) -> Dict

Extract weights for vanilla RNN.
Math: h_t = activation(W_x * x_t + W_h * h_{t-1} + b_x + b_h)
"""
function extract_vanilla_rnn_weights(layer::RNNLayerInfo)
    H = layer.hidden_size
    weights = Dict{String, Any}()
    
    # Julia W shape: [input_size, hidden_size, num_directions] (reversed from ONNX)
    # Julia R shape: [hidden_size, hidden_size, num_directions]
    
    # Forward direction - transpose to get [hidden_size, input_size] for W_x
    weights["W_x"] = permutedims(layer.W[:, :, 1])  # [hidden_size, input_size]
    weights["W_h"] = permutedims(layer.R[:, :, 1])  # [hidden_size, hidden_size]
    
    if !isnothing(layer.B)
        # Julia B shape: [2*hidden_size, num_directions] -> [Wb; Rb]
        weights["b_x"] = layer.B[1:H, 1]
        weights["b_h"] = layer.B[H+1:2H, 1]
    else
        weights["b_x"] = zeros(Float32, H)
        weights["b_h"] = zeros(Float32, H)
    end
    
    # Bidirectional weights (if applicable)
    if layer.num_directions == 2
        weights["W_x_backward"] = permutedims(layer.W[:, :, 2])
        weights["W_h_backward"] = permutedims(layer.R[:, :, 2])
        if !isnothing(layer.B)
            weights["b_x_backward"] = layer.B[1:H, 2]
            weights["b_h_backward"] = layer.B[H+1:2H, 2]
        else
            weights["b_x_backward"] = zeros(Float32, H)
            weights["b_h_backward"] = zeros(Float32, H)
        end
    end
    
    return weights
end

"""
    extract_lstm_weights(layer) -> Dict

Extract weights for LSTM with separate gate weights.

LSTM equations:
  i_t = σ(W_xi * x_t + W_hi * h_{t-1} + b_xi + b_hi)  # input gate
  f_t = σ(W_xf * x_t + W_hf * h_{t-1} + b_xf + b_hf)  # forget gate  
  c̃_t = tanh(W_xc * x_t + W_hc * h_{t-1} + b_xc + b_hc)  # cell candidate
  o_t = σ(W_xo * x_t + W_ho * h_{t-1} + b_xo + b_ho)  # output gate
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
  h_t = o_t ⊙ tanh(c_t)

ONNX LSTM weight order: [i, o, f, c] (differs from PyTorch [i, f, c, o]!)
"""
function extract_lstm_weights(layer::RNNLayerInfo)
    H = layer.hidden_size
    weights = Dict{String, Any}()
    
    # Julia W shape: [input_size, 4*hidden_size, num_directions] (reversed from ONNX)
    # ONNX gate order: i, o, f, c
    
    # Input weights (W_x for each gate) - transpose to [hidden_size, input_size]
    weights["W_xi"] = permutedims(layer.W[:, 1:H, 1])        # Input gate
    weights["W_xo"] = permutedims(layer.W[:, H+1:2H, 1])     # Output gate
    weights["W_xf"] = permutedims(layer.W[:, 2H+1:3H, 1])    # Forget gate
    weights["W_xc"] = permutedims(layer.W[:, 3H+1:4H, 1])    # Cell gate
    
    # Recurrent weights (W_h for each gate) - transpose to [hidden_size, hidden_size]
    weights["W_hi"] = permutedims(layer.R[:, 1:H, 1])
    weights["W_ho"] = permutedims(layer.R[:, H+1:2H, 1])
    weights["W_hf"] = permutedims(layer.R[:, 2H+1:3H, 1])
    weights["W_hc"] = permutedims(layer.R[:, 3H+1:4H, 1])
    
    # Biases
    if !isnothing(layer.B)
        # Julia B shape: [8*hidden_size, num_directions]
        # Order: Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c
        weights["b_xi"] = layer.B[1:H, 1]
        weights["b_xo"] = layer.B[H+1:2H, 1]
        weights["b_xf"] = layer.B[2H+1:3H, 1]
        weights["b_xc"] = layer.B[3H+1:4H, 1]
        
        weights["b_hi"] = layer.B[4H+1:5H, 1]
        weights["b_ho"] = layer.B[5H+1:6H, 1]
        weights["b_hf"] = layer.B[6H+1:7H, 1]
        weights["b_hc"] = layer.B[7H+1:8H, 1]
    else
        for gate in ["i", "o", "f", "c"]
            weights["b_x$gate"] = zeros(Float32, H)
            weights["b_h$gate"] = zeros(Float32, H)
        end
    end
    
    return weights
end

"""
    extract_gru_weights(layer) -> Dict

Extract weights for GRU with separate gate weights.

GRU equations:
  r_t = σ(W_xr * x_t + W_hr * h_{t-1} + b_xr + b_hr)  # reset gate
  z_t = σ(W_xz * x_t + W_hz * h_{t-1} + b_xz + b_hz)  # update gate
  h̃_t = tanh(W_xh * x_t + r_t ⊙ (W_hh * h_{t-1} + b_hh) + b_xh)  # candidate
  h_t = (1 - z_t) ⊙ h̃_t + z_t ⊙ h_{t-1}

ONNX GRU weight order: [z, r, h]
"""
function extract_gru_weights(layer::RNNLayerInfo)
    H = layer.hidden_size
    weights = Dict{String, Any}()
    
    # Julia W shape: [input_size, 3*hidden_size, num_directions] (reversed from ONNX)
    # ONNX gate order: z (update), r (reset), h (hidden/candidate)
    
    # Input weights - transpose to [hidden_size, input_size]
    weights["W_xz"] = permutedims(layer.W[:, 1:H, 1])        # Update gate
    weights["W_xr"] = permutedims(layer.W[:, H+1:2H, 1])     # Reset gate  
    weights["W_xh"] = permutedims(layer.W[:, 2H+1:3H, 1])    # Hidden/candidate gate
    
    # Recurrent weights - transpose to [hidden_size, hidden_size]
    weights["W_hz"] = permutedims(layer.R[:, 1:H, 1])
    weights["W_hr"] = permutedims(layer.R[:, H+1:2H, 1])
    weights["W_hh"] = permutedims(layer.R[:, 2H+1:3H, 1])
    
    # Biases
    if !isnothing(layer.B)
        # Julia B shape: [6*hidden_size, num_directions]
        # Order: Wb_z, Wb_r, Wb_h, Rb_z, Rb_r, Rb_h
        weights["b_xz"] = layer.B[1:H, 1]
        weights["b_xr"] = layer.B[H+1:2H, 1]
        weights["b_xh"] = layer.B[2H+1:3H, 1]
        
        weights["b_hz"] = layer.B[3H+1:4H, 1]
        weights["b_hr"] = layer.B[4H+1:5H, 1]
        weights["b_hh"] = layer.B[5H+1:6H, 1]
    else
        for gate in ["z", "r", "h"]
            weights["b_x$gate"] = zeros(Float32, H)
            weights["b_h$gate"] = zeros(Float32, H)
        end
    end
    
    return weights
end

# ============================================================================
# High-Level Analysis Functions
# ============================================================================

"""
    summarize_model(model::ParsedONNXModel)

Print a summary of the parsed ONNX model, showing only mathematical operations.
"""
function summarize_model(model::ParsedONNXModel)
    println("=" ^ 60)
    println("ONNX Model Summary (Mathematical Operations Only)")
    println("=" ^ 60)
    
    println("\n📊 Model Dimensions:")
    println("  Input size: $(model.model_input_size)")
    println("  Output size: $(model.model_output_size)")
    seq_str = isnothing(model.sequence_length) ? "dynamic" : string(model.sequence_length)
    println("  Sequence length: $seq_str")
    
    println("\n🔄 RNN Structure:")
    println("  Number of RNN modules: $(model.num_rnn_modules)")
    for (mod, layers) in sort(collect(model.layers_per_module))
        println("    Module $mod: $layers layer(s)")
    end
    
    if !isempty(model.rnn_layers)
        println("\n📋 RNN Layers:")
        for (i, layer) in enumerate(model.rnn_layers)
            println("  [$i] $(layer.name)")
            println("      Type: $(layer.op_type)")
            println("      Input size: $(layer.input_size)")
            println("      Hidden size: $(layer.hidden_size)")
            println("      Directions: $(layer.num_directions)")
            println("      Activations: $(join(layer.activations, ", "))")
        end
    end
    
    if !isempty(model.fc_layers)
        println("\n🔗 Fully Connected Layers:")
        for (i, layer) in enumerate(model.fc_layers)
            println("  [$i] $(layer.name) ($(layer.op_type))")
            println("      Input: $(layer.input_size) → Output: $(layer.output_size)")
        end
    end
    
    if !isempty(model.activations)
        println("\n⚡ Standalone Activations:")
        for (i, act) in enumerate(model.activations)
            println("  [$i] $(act.name): $(act.op_type)")
        end
    end
    
    println("\n📈 Computation Order (math ops only):")
    for (i, name) in enumerate(model.math_operation_order)
        println("  $i. $name")
    end
    
    println("\n" * "=" ^ 60)
end

"""
    get_weights_for_jump(model::ParsedONNXModel) -> Dict

Extract all weights in a format ready for JuMP constraint generation.
"""
function get_weights_for_jump(model::ParsedONNXModel)
    jump_weights = Dict{String, Any}()
    
    # RNN layers
    jump_weights["rnn_layers"] = Dict{Int, Dict}()
    for (i, layer) in enumerate(model.rnn_layers)
        jump_weights["rnn_layers"][i] = extract_rnn_weights(layer)
        jump_weights["rnn_layers"][i]["name"] = layer.name
        jump_weights["rnn_layers"][i]["type"] = layer.op_type
    end
    
    # FC layers
    jump_weights["fc_layers"] = Dict{Int, Dict}()
    for (i, layer) in enumerate(model.fc_layers)
        jump_weights["fc_layers"][i] = Dict(
            "name" => layer.name,
            "type" => layer.op_type,
            "W" => layer.W,
            "b" => isnothing(layer.b) ? zeros(Float32, layer.output_size) : layer.b,
            "input_size" => layer.input_size,
            "output_size" => layer.output_size
        )
    end
    
    # Model info
    jump_weights["model_info"] = Dict(
        "input_size" => model.model_input_size,
        "output_size" => model.model_output_size,
        "num_rnn_layers" => length(model.rnn_layers),
        "num_fc_layers" => length(model.fc_layers),
        "sequence_length" => model.sequence_length,
        "computation_order" => model.math_operation_order
    )
    
    return jump_weights
end

"""
    get_layer_connectivity(model::ParsedONNXModel) -> Dict

Analyze and return the connectivity between mathematical layers only.
"""
function get_layer_connectivity(model::ParsedONNXModel)
    connectivity = Dict{String, Dict{String, Any}}()
    
    # Map output names to their producing layer
    output_producers = Dict{String, String}()
    
    for layer in model.rnn_layers
        for out in layer.output_names
            output_producers[out] = layer.name
        end
    end
    
    for layer in model.fc_layers
        for out in layer.output_names
            output_producers[out] = layer.name
        end
    end
    
    for act in model.activations
        for out in act.output_names
            output_producers[out] = act.name
        end
    end
    
    # Build connectivity for RNN layers
    for layer in model.rnn_layers
        input_layers = String[]
        for inp in layer.input_names
            if haskey(output_producers, inp)
                push!(input_layers, output_producers[inp])
            end
        end
        connectivity[layer.name] = Dict(
            "type" => string(layer.op_type),
            "input_from" => input_layers,
            "output_to" => String[]
        )
    end
    
    # Build connectivity for FC layers
    for layer in model.fc_layers
        input_layers = String[]
        for inp in layer.input_names
            if haskey(output_producers, inp)
                push!(input_layers, output_producers[inp])
            end
        end
        connectivity[layer.name] = Dict(
            "type" => string(layer.op_type),
            "input_from" => input_layers,
            "output_to" => String[]
        )
    end
    
    # Build connectivity for activations
    for act in model.activations
        input_layers = String[]
        for inp in act.input_names
            if haskey(output_producers, inp)
                push!(input_layers, output_producers[inp])
            end
        end
        connectivity[act.name] = Dict(
            "type" => string(act.op_type),
            "input_from" => input_layers,
            "output_to" => String[]
        )
    end
    
    # Fill output_to by reversing input_from relationships
    for (layer_name, conn) in connectivity
        for input_layer in conn["input_from"]
            if haskey(connectivity, input_layer)
                push!(connectivity[input_layer]["output_to"], layer_name)
            end
        end
    end
    
    return connectivity
end

# ============================================================================
# Exports
# ============================================================================

export ParsedONNXModel, RNNLayerInfo, FCLayerInfo, ActivationInfo
export parse_onnx_graph, summarize_model
export extract_rnn_weights, get_weights_for_jump, get_layer_connectivity
export MATHEMATICAL_OPS, INFRASTRUCTURE_OPS, is_mathematical_op
