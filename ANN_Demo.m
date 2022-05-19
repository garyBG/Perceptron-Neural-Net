numEpochs = 25;
numHiddenNeurons = 10;
learningRate = 0.03;

% Use a script from Prof. Stiber for loading all MNIST data
rawImgs =loadMNISTImages("mnist/train-images.idx3-ubyte");
rawLabels=loadMNISTLabels("mnist/train-labels.idx1-ubyte");

testImgs = loadMNISTImages("mnist/t10k-images.idx3-ubyte");
testLabels = loadMNISTLabels("mnist/t10k-labels.idx1-ubyte");

% Process labels using MATLAB onehotencode
labels = (onehotencode(rawLabels, 2,"ClassNames",[0 1 2 3 4 5 6 7 8 9]))';
% Create network with 784 inputs, 10(?) hidden neurons (subject to change
% during testing), and 10 output neurons
mnNet=BackpropagationANN([784 numHiddenNeurons 10], @mySigmoid, @mySigmoidDerivative, learningRate);

% For storing mean square error to output a graph
mses = zeros(1, 500);

% Run through full training set for specified number of epochs
for epoch = 1:numEpochs
    for i=1:599
        iterMSE = mnNet.batchTrain(rawImgs(:,((i-1)*100) + 1 : (i*100)), labels(:,((i-1)*100) + 1 : (i*100)));
        mses(1, i) = iterMSE;
    %     disp("MSE: " + iterMSE);
    end
    numCorrect = 0;
    numTested = 0;
    disp("Epoch #" + epoch + " completed.");
    % Full run through test set to monitor performance of network
    % post-epoch training
    for col = 1:size(testImgs,2)
        out = mnNet.forward(testImgs(:,col));
        if (find(max(out) == out) - 1) == testLabels(col)
            numCorrect = numCorrect + 1;
        end
        numTested = numTested + 1;
    end
    disp((numCorrect / numTested) * 100 + "% correct on test set.");
end
% disp(round(mnNet.forward(rawImgs(:,1))));