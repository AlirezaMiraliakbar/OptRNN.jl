module OptRNN

# Include submodules and core functionality
# include("types/RNN.jl")
# include("types/LSTM.jl")
# include("types/GRU.jl")

include("utils.jl")

# Export main types and functions

export register_RNN!, register_LSTM!, register_GRU!, load_onnx

# Package version
const VERSION = v"0.1.0"

"""
    OptRNN

A Julia package for embedding trained Recurrent Neural Networks (RNNs).

"""


function register_RNN!()

end

function register_LSTM!()
    error("LSTM registration not implemented yet!")
    return nothing
end

function register_GRU!()
    error("GRU registration not implemented yet!")
    return nothing
end

end # module OptRNN