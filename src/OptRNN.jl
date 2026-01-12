module OptRNN

# Include submodules and core functionality
# include("types/RNN.jl")
# include("types/LSTM.jl")
# include("types/GRU.jl")

include("utils.jl")

using JuMP

# Export main types and functions

export register_RNN!, register_LSTM!, register_GRU!, PytorchModel, ONNXModel

# Package version
const VERSION = v"0.1.0"

"""
    OptRNN

A Julia package for embedding trained Recurrent Neural Networks (RNNs).

"""


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
register_RNN!(model; model_type=:onnx, full_space=true)
register_RNN!(model; model_type=:pytorch, reduced_space=true)
register_RNN!(model; model_type=:onnx, hybrid=true)
```
"""
function register_RNN!(model::JuMP.AbstractModel; model_type::Symbol=:onnx, full_space::Bool=false, reduced_space::Bool=false, hybrid::Bool=false)
    # Validate model_type
    if model_type !== :onnx && model_type !== :pytorch
        error("model_type must be either :onnx or :pytorch. Got: $model_type")
    end
    
    # Validate that exactly one method is specified
    count_true = sum([full_space, reduced_space, hybrid])
    if count_true != 1
        error("Exactly one of full_space, reduced_space, or hybrid must be set to true. Got: full_space=$full_space, reduced_space=$reduced_space, hybrid=$hybrid")
    end
    
    # Dispatch to appropriate method based on keyword
    if full_space
        return _register_RNN!(model, model_type, Val(:full_space))
    elseif reduced_space
        return _register_RNN!(model, model_type, Val(:reduced_space))
    elseif hybrid
        return _register_RNN!(model, model_type, Val(:hybrid))
    end
end

# Multiple dispatch methods for different formulations
function _register_RNN!(model::JuMP.AbstractModel, model_type::Symbol, ::Val{:full_space})
    # Full space formulation implementation
    # TODO: Implement full space formulation
    @info "Registering RNN with full_space formulation using $model_type model type"
    error("Full space formulation not implemented yet!")
end

function _register_RNN!(model::JuMP.AbstractModel, model_type::Symbol, ::Val{:reduced_space})
    # Reduced space formulation implementation
    # TODO: Implement reduced space formulation
    @info "Registering RNN with reduced_space formulation using $model_type model type"
    error("Reduced space formulation not implemented yet!")
end

function _register_RNN!(model::JuMP.AbstractModel, model_type::Symbol, ::Val{:hybrid})
    # Hybrid formulation implementation
    # TODO: Implement hybrid formulation
    @info "Registering RNN with hybrid formulation using $model_type model type"
    error("Hybrid formulation not implemented yet!")
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

end # module OptRNN