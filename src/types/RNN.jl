using JuMP
using PythonCall

include("../utils.jl")

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

function register_RNN!(model::JuMP.Model, reg_method::String = "HB")

    if reg_method == ["hybrid", "reduced"] 
        error("$reg_method formulation not supported yet!")
    else
        
        # implemening full_space formulation regiteration 
        register_full(model, onnx_params)
    end
end