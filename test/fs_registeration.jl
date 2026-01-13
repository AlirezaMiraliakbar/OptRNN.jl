using OptRNN
using JuMP
using Ipopt 
using Revise

current_dir = @__DIR__
model = Model(Ipopt.Optimizer)
ml_path = joinpath(current_dir, "models/rnn_model.pth")

ml_model = PyTorchModel(ml_path)

opt_model = Model(Ipopt.Optimizer)

@variable(opt_model, -1 <= x[1:10] <= 1) # we need to link x 
