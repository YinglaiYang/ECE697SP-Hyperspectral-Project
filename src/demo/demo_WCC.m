clc, clear, close all;

load hsi;
load mapTrain;
load mapTest;
load trainingSamplesPerClass;

%% Parameters
alpha = 0.5;
sigma = 0.1;

nystroemFraction = 1e-3; % 1percent
 
width = 100;
height = 100;

RSVD.k_fraction = 0.7;
RSVD.p = 10;
RSVD.q = 3;

testIDX = mapTest > 0;

tic;
%% Perform classification
predictedLabels = windowedClassistClassifier(hsi, mapTrain, testIDX, alpha, sigma, nystroemFraction, RSVD, width, height);

mapPredict = reshape(predictedLabels, size(mapTrain));

%% Error rate
testError = errorRate(mapPredict, mapTest);
confusionMatrix = getConfusionMatrix(mapPredict, mapTest);
[precision, recall] = precisionRecall(confusionMatrix);

toc

showMap(mapPredict);