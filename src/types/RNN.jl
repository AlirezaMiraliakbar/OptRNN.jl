
struct RNNModel <: NNModel 
    seq_length::Int64;
    input_size::Int64;
    hidden_size::Array{Int64};
    output_size::Int64;
    Wx::Array{Float64,2};
    Wh::Array{Float64,2};
    bx::Vector{Float64};
    bh::Vector{Float64};
    act_func::ActFunc;
    direction::Int64;
end

# translating the written PyTorch model to RNNModel structure

# Multiple dispatch methods for different formulations
function _register_RNN!(model::JuMP.AbstractModel, rnn_model::PyTorchModel, ::Val{:full_space})
    # Full space formulation implementation
    # TODO: Implement full space formulation
    @info "Registering RNN with full_space formulation using PyTorch model type"

    error("Full space formulation not implemented yet!")
end

function _register_RNN!(model::JuMP.AbstractModel, rnn_model::PyTorchModel, ::Val{:reduced_space})
    # Reduced space formulation implementation
    # TODO: Implement reduced space formulation
    @info "Registering RNN with reduced_space formulation using PyTorch model type"
    error("Reduced space formulation not implemented yet!")
end

function _register_RNN!(model::JuMP.AbstractModel, rnn_model::PyTorchModel, ::Val{:hybrid})
    # Hybrid formulation implementation
    # TODO: Implement hybrid formulation
    @info "Registering RNN with hybrid formulation using PyTorch model type"
    error("Hybrid formulation not implemented yet!")
end

# Multiple dispatch methods for different formulations for ONNXModel
function _register_RNN!(model::JuMP.AbstractModel, rnn_model::ONNXModel, ::Val{:full_space})
    # Full space formulation implementation
    # TODO: Implement full space formulation
    @info "Registering RNN with full_space formulation using PyTorch model type"

    error("Full space formulation not implemented yet!")
end

function _register_RNN!(model::JuMP.AbstractModel, rnn_model::ONNXModel, ::Val{:reduced_space})
    # Reduced space formulation implementation
    # TODO: Implement reduced space formulation
    @info "Registering RNN with reduced_space formulation using PyTorch model type"
    error("Reduced space formulation not implemented yet!")
end

function _register_RNN!(model::JuMP.AbstractModel, rnn_model::ONNXModel, ::Val{:hybrid})
    # Hybrid formulation implementation
    # TODO: Implement hybrid formulation
    @info "Registering RNN with hybrid formulation using PyTorch model type"
    error("Hybrid formulation not implemented yet!")
end

function RNNModel(model_load_path::String)
    # TODO: Add implementation to load model parameters from file
    # This would typically involve reading the file and extracting:
    # seq_length, hidden_size, input_size, output_size, Wx, Wh, bx, bh, act_func
    onnx_model = ONNXLoader(model_load_path)
    # the required RNN properties will be retrived from the ONNXModel structure that is built 
    seq_length = 0
    input_size = onnx_model.input_size
    hidden_sizes = []

    output_size = onnx_model.output_size
    
    return RNNModel(seq_length, input_size, hidden_size, output_size, Wx, Wh, bx, bh, act_func)
end

# Alternative constructor with default activation function
function RNNModel(seq_length::Int64, hidden_size::Int64, input_size::Int64, output_size::Int64, 
                  Wx::Array{Float64,2}, Wh::Array{Float64,2}, bx::Vector{Float64}, bh::Vector{Float64})
    return RNNModel(seq_length, hidden_size, input_size, output_size, Wx, Wh, bx, bh, tanh)
end