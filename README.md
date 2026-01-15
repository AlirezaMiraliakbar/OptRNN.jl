# OptRNN.jl
A Julia package developed to register recurrent-based neural networks directly to a JuMP model.

## Requirments
OptRNN works on ONNX exported models. If using PyTorch to export your model, make sure to set `dynamo = false` when 
exporting to ONNX. For example:
```python
# DO NOT use dynamo=True (which is the new default in some contexts)
# DO use standard torch.onnx.export
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    opset_version=14  # Version 14+ has excellent LSTM support
)
```

## Example
Here is an simple example on how OptRNN can work to register a model using its full-space formulation registeration:

```julia 
using OptRNN
using JuMP
using EAGO 


model = Model(EAGO.Optimizer)
ml_model = ONNXModel('rnn_model.onnx')

register_RNN!(model, ml_model; full_space = true)
```
