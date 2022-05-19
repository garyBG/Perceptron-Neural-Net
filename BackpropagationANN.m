classdef BackpropagationANN < handle
    %BackpropagationANN multi-layer, backpropagation-trained artificial
    %neural network
    
    properties
        layerWeights
        transFn
        dTransFn
        learningRate
    end
    
    methods
        function obj = BackpropagationANN(layerDims, transFn, dTransFn, ...
                learningRate)
            %BackpropagationANN.m multi-layer network implementing
            %backpropagation.
            % Inputs:
            %   layerDims: an array of dimensions, one integer value per
            %   layer (including input layer and output layer) from which
            %   to build the multi-layer network. -must- contain at least 2
            %   items
            %
            %   transFn: transfer function to use on all layers
            %
            %   dTransFn: derivative of the transfer function
            %
            %   learningRate: alpha value to use as a weight on learning
            
            % weightInitEpsilon is a weight initialization hyperparameter
            weightInitEpsilon = 0.1;
%             weightInitEpsilon = 1/786;
            
            obj.transFn = transFn;
            obj.dTransFn = dTransFn;
            obj.learningRate = learningRate;
            
            % Multilayer network needs one matrix per layer
            % layerWeights contains a cell array with 1 row containing
            % columns of weight matrices, one per set of weights between
            % layers
            obj.layerWeights = cell(1, size(layerDims, 2) - 1);
            for layerNum = 1:(size(layerDims, 2) - 1)
                %rand function gives random uniform values for the
                %specified dimensions.
                
                %Weight matrix dimensions for each layer are (output,input)
                %In addition, the "input" dimension is expanded by 1 to
                %incorporate the bias.
                W = rand(layerDims(layerNum + 1), layerDims(layerNum) + 1);
                % At the moment, make the weights in each layer small
                % random values from -0.2 to 0.2
                W = W * 2;
                W = W - 1;
                W = W * weightInitEpsilon;
                
                obj.layerWeights{layerNum} = W;
            end
            
        end
        function mse = batchTrain(obj, inputs, targets)
            % Inputs:
            %   inputs: all inputs for the training set formatted as one
            %   input per column
            %
            %   targets: all intended outputs for the inputs, one per
            %   column
            % Output:
            %   mse: set of mean square errors during forward propagation
            
            
            numWeightedLayers = size(obj.layerWeights, 2);
            curOutputs = inputs;
            
            % allNets saves all net layer outputs
            allNets = cell(1, numWeightedLayers);
            % allOutputs saves all layer outputs, with transfer function
            % applied. ALSO its first entry is the input to allow weight
            % and bias updates to work seamlessly
            allOutputs = cell(1, numWeightedLayers + 1);
            allOutputs{1} = inputs;
            

%-----------------------Forward propagation begins------------------------
            for layerNum = 1: numWeightedLayers
                curLayerWeights = obj.layerWeights{layerNum};
                
                % Append another ones row for the output at each layer
                % preceding the algorithm of transfer function applied to
                % weights times inputs (plus implied bias)
                
                % Save net inputs to the transfer function for use in
                % sensitivity calculation
                curNets = curLayerWeights * ...
                    [curOutputs ; ones(1, size(curOutputs, 2))];
                curOutputs = obj.transFn(curNets);
                
                allNets{layerNum} = curNets;
                allOutputs{layerNum + 1} = curOutputs;
            end
%---------------------Sensitivity calculations begin----------------------
            % s^M = -2 dF(n^m).*(t-a^M)
            errors = targets - curOutputs;
            % Output mean square errors
            mse = mean2(errors.^2);
            
            outputSensitivities = -2 * ...
                obj.dTransFn(allNets{size(allNets, 2)}) .* ...
                errors;
            
            % Store layer sensitivities in one cell array per set of
            % weights
            sensitivities = cell(1, numWeightedLayers);
            sensitivities{numWeightedLayers} = outputSensitivities;
            % Count layer numbers backwards from the second to last layer,
            % since the last layer's sensitivities are already calculated
            for layerNum = (numWeightedLayers - 1) : -1 : 1
                % Strip the biases to work with only the weights, by
                % selecting only the columns up to, but not including, the
                % ending column in the layer weights
                sensitivities{layerNum} = ...
                    obj.dTransFn(allNets{layerNum}) .* ...
                    (obj.layerWeights{layerNum + 1} ...
                    (:,1:size(obj.layerWeights{layerNum + 1}, 2) - 1)' * ...
                    sensitivities{layerNum + 1});
            end
%-----------------------Weight and bias updates begin---------------------
            
            % Iterate through updating weights backwards
            for layerNum = numWeightedLayers: -1 : 1
                % For the moment I'm having trouble figuring out how to do
                % a single matrix operation during batch learning, so
                % instead, work with the sensitivities and outputs one at a
                % time in updating the weight matrix

                for trainingNumber = 1: size(sensitivities{layerNum}, 2)
                    % Recall that allOutputs starts with the input
                    obj.layerWeights{layerNum} = ...
                        obj.layerWeights{layerNum} - (obj.learningRate ...
                        * sensitivities{layerNum}(:,trainingNumber) * ...
                        [allOutputs{layerNum}(:,trainingNumber) ; 1]'); 
                end
            end
        end
        function output = forward(obj,input)
        %Forward propagation implementation
        % Input:
        %   input: single input column to run through network   
            output = input; % Output is the running output through layers
            for Wn = 1: size(obj.layerWeights, 2)
                % Append 1 to the current output to account for bias
                output = obj.transFn(obj.layerWeights{Wn} * [output ; 1]);
            end
        end

    end
end

