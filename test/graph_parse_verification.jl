# this test file is intented to test the graph verification functionalities
# of the OptRNN package

using OptRNN
using Test

current_dir = @__DIR__

println("Running OptRNN graph verification tests in directory: $current_dir")

model_path = joinpath(current_dir, "models", "rnn_model.onnx")

@testset "ONNX Graph Parsing" begin
    
    @testset "Model Loading" begin
        # Test that parse_onnx_graph returns a ParsedONNXModel
        onnx_model = parse_onnx_graph(model_path)
        @test onnx_model isa ParsedONNXModel
        
        # Test that the raw graph is preserved
        @test onnx_model.raw_graph isa ONNXLowLevel.GraphProto
    end
    
    @testset "Model Structure" begin
        onnx_model = parse_onnx_graph(model_path)
        
        # Test that we have RNN layers detected
        @test length(onnx_model.rnn_layers) == 2
        @test onnx_model.num_rnn_modules == 1
        
        # Test layers_per_module is consistent with rnn_layers count
        total_layers_from_dict = sum(values(onnx_model.layers_per_module))
        @test total_layers_from_dict == length(onnx_model.rnn_layers)
        
        # Test model dimensions are positive
        @test onnx_model.model_input_size == 3
        @test onnx_model.model_output_size == 2
    end
    
    @testset "RNN Layer Properties" begin
        onnx_model = parse_onnx_graph(model_path)
        
        for (i, rnn_layer) in enumerate(onnx_model.rnn_layers)
            @testset "RNN Layer $i: $(rnn_layer.name)" begin
                # Test layer info types
                @test rnn_layer isa RNNLayerInfo
                @test rnn_layer.name isa String
                @test !isempty(rnn_layer.name)
                
                # Test op_type is valid RNN type
                @test rnn_layer.op_type in [:RNN, :LSTM, :GRU]
                
                # Test dimensions are positive
                @test rnn_layer.input_size > 0
                @test rnn_layer.hidden_size > 0
                @test rnn_layer.num_directions in [1, 2]  # Unidirectional or bidirectional
                
                # Test layer/module indices are non-negative
                @test rnn_layer.layer_index >= 0
                @test rnn_layer.module_index >= 1
                
                # Test weight matrices exist and have correct dimensions
                # Julia shape (reversed from ONNX): [input_size, hidden_size*, num_directions]
                @test size(rnn_layer.W, 3) == rnn_layer.num_directions
                @test size(rnn_layer.R, 3) == rnn_layer.num_directions
                @test size(rnn_layer.R, 1) == size(rnn_layer.R, 2)  # R is square (hidden_size x hidden_size) in first 2 dims
                
                # Test weight dimensions based on RNN type
                hidden_mult = if rnn_layer.op_type == :RNN
                    1
                elseif rnn_layer.op_type == :LSTM
                    4
                elseif rnn_layer.op_type == :GRU
                    3
                end
                expected_w_dim2 = hidden_mult * rnn_layer.hidden_size
                @test size(rnn_layer.W, 2) == expected_w_dim2
                @test size(rnn_layer.W, 1) == rnn_layer.input_size  # input_size is first dim in Julia
                
                # Test input/output names are present
                @test length(rnn_layer.input_names) > 0
                @test length(rnn_layer.output_names) > 0
                
                # Test activations are present
                @test length(rnn_layer.activations) > 0
                @test all(act -> act isa String, rnn_layer.activations)
            end
        end
    end
    
    @testset "FC Layer Properties" begin
        onnx_model = parse_onnx_graph(model_path)
        
        for (i, fc_layer) in enumerate(onnx_model.fc_layers)
            @testset "FC Layer $i: $(fc_layer.name)" begin
                @test fc_layer isa FCLayerInfo
                @test fc_layer.name isa String
                @test !isempty(fc_layer.name)
                
                # Test op_type is valid
                @test fc_layer.op_type in [:Gemm, :MatMul]
                
                # Test dimensions are positive
                @test fc_layer.input_size > 0
                @test fc_layer.output_size > 0
                
                # Test weight matrix dimensions
                @test size(fc_layer.W) == (fc_layer.input_size, fc_layer.output_size) || 
                      size(fc_layer.W) == (fc_layer.output_size, fc_layer.input_size)
                
                # Test bias if present
                if !isnothing(fc_layer.b)
                    @test length(fc_layer.b) == fc_layer.output_size
                end
                
                # Test input/output names
                @test length(fc_layer.input_names) > 0
                @test length(fc_layer.output_names) > 0
            end
        end
    end
    
    @testset "Mathematical Operations Only" begin
        onnx_model = parse_onnx_graph(model_path)
        
        # Test that only mathematical operations are in computation order
        for op_name in onnx_model.math_operation_order
            @test op_name isa String
            @test !isempty(op_name)
        end
        
        # Test that infrastructure ops are NOT in the math_operation_order
        infrastructure_keywords = ["Transpose", "Reshape", "Squeeze", "Unsqueeze", 
                                   "Shape", "Gather", "Concat", "Constant", "Identity"]
        for op_name in onnx_model.math_operation_order
            op_lower = lowercase(op_name)
            for infra_kw in infrastructure_keywords
                # Operation names shouldn't start with infrastructure op types
                @test !startswith(op_lower, lowercase(infra_kw))
            end
        end
    end
    
    @testset "Input/Output Info" begin
        onnx_model = parse_onnx_graph(model_path)
        
        # Test input info
        @test length(onnx_model.input_info) > 0
        for input in onnx_model.input_info
            @test haskey(input, "name")
            @test haskey(input, "shape")
            @test input["name"] isa String
            @test input["shape"] isa Vector
        end
        
        # Test output info
        @test length(onnx_model.output_info) > 0
        for output in onnx_model.output_info
            @test haskey(output, "name")
            @test haskey(output, "shape")
            @test output["name"] isa String
            @test output["shape"] isa Vector
        end
    end
    
    @testset "Weight Extraction for JuMP" begin
        onnx_model = parse_onnx_graph(model_path)
        
        # Test extract_rnn_weights for each RNN layer
        for (i, rnn_layer) in enumerate(onnx_model.rnn_layers)
            weights = extract_rnn_weights(rnn_layer)
            
            @testset "Weights for RNN Layer $i" begin
                # Common fields
                @test haskey(weights, "hidden_size")
                @test haskey(weights, "input_size")
                @test haskey(weights, "num_directions")
                @test haskey(weights, "activations")
                
                @test weights["hidden_size"] == rnn_layer.hidden_size
                @test weights["input_size"] == rnn_layer.input_size
                
                # Type-specific weights
                if rnn_layer.op_type == :RNN
                    @test haskey(weights, "W_x")
                    @test haskey(weights, "W_h")
                    @test haskey(weights, "b_x")
                    @test haskey(weights, "b_h")
                    @test size(weights["W_x"]) == (rnn_layer.hidden_size, rnn_layer.input_size)
                    @test size(weights["W_h"]) == (rnn_layer.hidden_size, rnn_layer.hidden_size)
                    
                elseif rnn_layer.op_type == :LSTM
                    # LSTM has 4 gates: i, o, f, c
                    for gate in ["i", "o", "f", "c"]
                        @test haskey(weights, "W_x$gate")
                        @test haskey(weights, "W_h$gate")
                        @test haskey(weights, "b_x$gate")
                        @test haskey(weights, "b_h$gate")
                    end
                    
                elseif rnn_layer.op_type == :GRU
                    # GRU has 3 gates: z, r, h
                    for gate in ["z", "r", "h"]
                        @test haskey(weights, "W_x$gate")
                        @test haskey(weights, "W_h$gate")
                        @test haskey(weights, "b_x$gate")
                        @test haskey(weights, "b_h$gate")
                    end
                end
            end
        end
        
        # Test get_weights_for_jump
        jump_weights = get_weights_for_jump(onnx_model)
        
        @test haskey(jump_weights, "rnn_layers")
        @test haskey(jump_weights, "fc_layers")
        @test haskey(jump_weights, "model_info")
        
        @test jump_weights["model_info"]["input_size"] == onnx_model.model_input_size
        @test jump_weights["model_info"]["output_size"] == onnx_model.model_output_size
        @test jump_weights["model_info"]["num_rnn_layers"] == length(onnx_model.rnn_layers)
        @test jump_weights["model_info"]["num_fc_layers"] == length(onnx_model.fc_layers)
    end
    
    @testset "Layer Connectivity" begin
        onnx_model = parse_onnx_graph(model_path)
        connectivity = get_layer_connectivity(onnx_model)
        
        # Test connectivity structure
        @test connectivity isa Dict
        
        # All RNN layers should be in connectivity
        for rnn_layer in onnx_model.rnn_layers
            @test haskey(connectivity, rnn_layer.name)
            conn = connectivity[rnn_layer.name]
            @test haskey(conn, "type")
            @test haskey(conn, "input_from")
            @test haskey(conn, "output_to")
            @test conn["type"] == string(rnn_layer.op_type)
        end
        
        # All FC layers should be in connectivity
        for fc_layer in onnx_model.fc_layers
            @test haskey(connectivity, fc_layer.name)
            conn = connectivity[fc_layer.name]
            @test haskey(conn, "type")
            @test haskey(conn, "input_from")
            @test haskey(conn, "output_to")
        end
    end
    
    @testset "Summary Function" begin
        onnx_model = parse_onnx_graph(model_path)
        
        # Test that summarize_model runs without error
        @test begin
            summarize_model(onnx_model)
            true
        end
    end
end

println("\n✅ All graph verification tests completed!")
