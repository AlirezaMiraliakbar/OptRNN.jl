module OptRNN

using JuMP
using Reexport
using ONNXLowLevel
# Export main types and functions
export register_RNN!, register_LSTM!, register_GRU!, PyTorchModel, ONNXModel
export ParsedONNXModel, RNNLayerInfo, FCLayerInfo, ActivationInfo
export parse_onnx_graph, summarize_model
export extract_rnn_weights, get_weights_for_jump, get_layer_connectivity
export MATHEMATICAL_OPS, INFRASTRUCTURE_OPS, is_mathematical_op

# Package version
const VERSION = v"0.1.0"

"""
    OptRNN

A Julia package for embedding trained Recurrent Neural Networks (RNNs).

"""

abstract type NNModel end
abstract type ActFunc end


include("types/parse/pytorch.jl")
include("types/parse/onnx.jl")
include("types/parse/onnx_parser.jl")
include("types/RNN.jl")
include("types/LSTM.jl")
include("types/GRU.jl")
include("utils.jl")

# Public API with keyword arguments
"""
    register_RNN!(model::JuMP.AbstractModel; model_type::Symbol=:onnx, full_space::Bool=false, reduced_space::Bool=false, hybrid::Bool=false)

Register an RNN to a JuMP model using the specified formulation method.

# Arguments
- `model::JuMP.AbstractModel`: The JuMP model to register the RNN to
- `model_type::Symbol`: The type of model reference - `:onnx` for ONNX model or `:pytorch` for PyTorch saved model (default: `:onnx`)
- `full_space::Bool`: Use full space formulation (default: false)
- `reduced_space::Bool`: Use reduced space formulation (default: false)
- `hybrid::Bool`: Use hybrid formulation (default: false)

# Examples
```julia
register_RNN!(model; full_space=true)
register_RNN!(model; reduced_space=true)
register_RNN!(model; hybrid=true)
```
```
"""
function register_RNN!(model::JuMP.AbstractModel, rnn_model::PyTorchModel; full_space::Bool=false, reduced_space::Bool=false, hybrid::Bool=false)
    
    println("Registering given PyTorch RNN model...")

    # Validate that exactly one method is specified
    count_true = sum([full_space, reduced_space, hybrid])
    if count_true != 1
        error("Exactly one of full_space, reduced_space, or hybrid must be set to true. Got: full_space=$full_space, reduced_space=$reduced_space, hybrid=$hybrid")
    end
    
    # Dispatch to appropriate method based on keyword
    if full_space
        return _register_RNN!(model, rnn_model, Val(:full_space))
    elseif reduced_space
        return _register_RNN!(model, rnn_model, Val(:reduced_space))
    elseif hybrid
        return _register_RNN!(model, rnn_model, Val(:hybrid))
    end
end

function register_RNN!(model::JuMP.AbstractModel, rnn_model::ONNXModel; full_space::Bool=false, reduced_space::Bool=false, hybrid::Bool=false)
    
    println("Registering given ONNX RNN model...")
    # Validate that exactly one method is specified
    count_true = sum([full_space, reduced_space, hybrid])
    if count_true != 1
        error("Exactly one of full_space, reduced_space, or hybrid must be set to true. Got: full_space=$full_space, reduced_space=$reduced_space, hybrid=$hybrid")
    end
    
    # Dispatch to appropriate method based on keyword
    if full_space
        return _register_RNN!(model, rnn_model, Val(:full_space))
    elseif reduced_space
        return _register_RNN!(model, rnn_model, Val(:reduced_space))
    elseif hybrid
        return _register_RNN!(model, rnn_model, Val(:hybrid))
    end
end
# ===============================================================================================================================================================================================================================================
# TODO: implement LSTM and GRU later
function register_LSTM!()
    error("LSTM registration not implemented yet!")
    return nothing
end

function register_GRU!()
    error("GRU registration not implemented yet!")
    return nothing
end

# Re-export dependencies for convenient REPL usage
@reexport using JuMP
@reexport using ONNXLowLevel
end # module OptRNN