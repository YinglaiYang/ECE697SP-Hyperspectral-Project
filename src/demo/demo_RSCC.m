clc, clear, close all;

load hsi;
load mapTrain;
load mapTest;
load trainingSamplesPerClass;

%% Parameters
alpha = 0.01;
sigma = 0.01;

nystroemFraction = 1e-3; % 1percent

sectionSize = 10000;

RSVD.k_fraction = 0.7;
RSVD.p = 10;
RSVD.q = 3;

testIDX = mapTest > 0;

tic;
%% Perform classification
predictedLabels = randomizedSetsClassistClassifier(hsi, mapTrain, testIDX, alpha, sigma, nystroemFraction, RSVD, sectionSize);

mapPredict = reshape(predictedLabels, size(mapTrain));

%% Error rate
testError = errorRate(mapPredict, mapTest);
confusionMatrix = getConfusionMatrix(mapPredict, mapTest);
[precision, recall] = precisionRecall(confusionMatrix);

toc

showMap(mapPredict);