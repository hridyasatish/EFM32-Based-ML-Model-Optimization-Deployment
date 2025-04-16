
% This script trains a digit classification CNN, prunes it iteratively using L1 norm,
and performs post-training quantization. Visualizations and model statistics are also included.

% -----------------------------
% Step 1: Load Dataset & Train
% -----------------------------
[imdsTrain, imdsValidation] = loadDigitDataset;
net = trainDigitDataNetwork(imdsTrain, imdsValidation);
trueLabels = imdsValidation.Labels;
classes = categories(trueLabels);

executionEnvironment = "auto";
miniBatchSize = 128;
imdsValidation.ReadSize = miniBatchSize;
mbqValidation = minibatchqueue(imdsValidation,1,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFormat','SSCB',...
    'MiniBatchFcn',@preprocessMiniBatch,...
    'OutputEnvironment',executionEnvironment);

lgraph = layerGraph(net.Layers);
lgraph = removeLayers(lgraph,["softmax","classoutput"]);
dlnetog = dlnetwork(lgraph);
accuracyOriginalNet = evaluateAccuracy(dlnetog,mbqValidation,classes,trueLabels);

% -----------------------------
% Step 2: Initialize Pruning
% -----------------------------
load('digitsNet.mat','net')
convIndices = findConvLayers(net.Layers);
bnIndices = findBatchNormLayers(net.Layers);
fcIndex = findFCLayers(net.Layers);

prune_ratio = 0.1;
prune_iterations = int32(1/prune_ratio)-1;
prunedChannelsPerItr = zeros(length(convIndices), prune_iterations);
numOutChannelsPerLayer = zeros(length(convIndices),1);

for i=1:length(convIndices)
    layer = net.Layers(convIndices(i));
    if isa(layer, 'nnet.cnn.layer.Convolution2DLayer')
        numOutChannelsPerLayer(i) = layer.NumFilters;
    end
end

prunedAccuracies = zeros(prune_iterations, 1);
sparsityLevels = zeros(prune_iterations, 1);
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, 'MaxEpochs',10, ...
    'Shuffle','every-epoch', 'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, 'Verbose',false, ...
    'Plots','none',"ExecutionEnvironment","auto");

iter=1;
while true
    [convWeights, ~] = getConvWeights(net, convIndices);
    pruneFilters = computeL1Pruning(convWeights, prune_ratio);
    if iter-1 >= prune_iterations
        break;
    end
    [net,prunedChannelsPerItr(:,iter)]= pruneNetwork(net, convIndices, bnIndices, fcIndex, pruneFilters);
    lgraph_1 = layerGraph(net.Layers);
    prunedNet = trainNetwork(imdsTrain, lgraph_1, options);
    lgraph_2 = layerGraph(prunedNet.Layers);
    lgraph_2 = removeLayers(lgraph_2, ["softmax","classoutput"]);
    dlnet_2 = dlnetwork(lgraph_2);
    prunedAccuracies(iter) = evaluateAccuracy(dlnet_2,mbqValidation,classes,trueLabels);
    sparsityLevels(iter) = iter * prune_ratio * 100;
    iter=iter+1;
end

% -----------------------------
% Step 3: Plot Pruning Accuracy
% -----------------------------
figure
plot(sparsityLevels, prunedAccuracies*100, '-og','LineWidth',2,'MarkerSize',6);
xlabel('Sparsity (%)'); ylabel('Accuracy (%)');
title('Pruning Accuracy Trend'); grid on;

% -----------------------------
% Step 4: Plot Layer-wise Filters
% -----------------------------
prunedChannelsPerLayer = [sum(prunedChannelsPerItr, 2);0];
remainingData = [numOutChannelsPerLayer-prunedChannelsPerLayer(1:3); size(net.Layers(fcIndex).Weights, 1)];
if remainingData(end) == 0; remainingData(end) = 1; end
layerNames = arrayfun(@(x) x.Name, net.Layers(convIndices), 'UniformOutput', false);
fcLayerName = net.Layers(fcIndex).Name;
selectedLayerNames = [layerNames; {fcLayerName}];
figure
bar([prunedChannelsPerLayer,remainingData],"stacked")
xlabel("Layer"); ylabel("Number of filters");
title("Number of Filters per Layer")
xticks(1:(numel(selectedLayerNames)))
xticklabels(selectedLayerNames)
xtickangle(45)
legend("Pruned","Remaining","Location","southoutside")
set(gca,'TickLabelInterpreter','none')

% -----------------------------
% Step 5: Quantization
% -----------------------------
calibrationDataStore = splitEachLabel(imdsTrain,0.1,'randomize');
validationDataStore = imdsValidation;
load('digitsNet_0.90_sparsity_params_6634.mat','net')
quantObjPrunedNetwork = dlquantizer(net,'ExecutionEnvironment','GPU');
quantOpts = dlquantizationOptions('Target','host');
calResults = calibrate(quantObjPrunedNetwork, calibrationDataStore);
valResults = validate(quantObjPrunedNetwork, validationDataStore, quantOpts);
valResults.MetricResults.Result
save('quantObjPrunedNetworkCalResults.mat','calResults');
save('quantObjPrunedNetwork.mat','quantObjPrunedNetwork');

% End of Script


