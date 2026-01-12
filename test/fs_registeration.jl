using OptRNN
using JuMP
using Ipopt 
using Revise

current_dir = @__DIR__
model = Model(Ipopt.Optimizer)
ml_path = joinpath(current_dir, "models/rnn_model.pth")

ml_model = PytorchModel(ml_path)
println("Loaded PyTorch model: ", ml_model)
