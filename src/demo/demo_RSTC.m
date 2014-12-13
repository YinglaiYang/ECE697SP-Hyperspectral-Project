clc, clear, close all;

load hsi;
load mapTrain;
load mapTest;
load trainingSamplesPerClass;

% %% Test
% lots = [2, 3, 6, 11, 14, 15, 19];
% 
% mapTrain(~ismember(mapTrain, lots)) = 0;
% mapTest(~ismember(mapTest, lots)) = 0;

%% Parameters
alpha = 0.5;
sigma = 0.25;

nystroemFraction = 1e-3; % 1percent

sectionSize = 10000;

RSVD.k_fraction = 0.7;
RSVD.p = 10;
RSVD.q = 3;

testIDX = mapTest > 0;

tic;
%% Perform classification
predictedLabels = randomizedSetsClassifier(hsi, mapTrain, alpha, sigma, nystroemFraction, RSVD, sectionSize);

mapPredict = reshape(predictedLabels, size(mapTrain));

%% Error rate
testError = errorRate(mapPredict, mapTest);
cm = getConfusionMatrix(mapPredict, mapTest);
[precision, recall] = precisionRecall(cm);

toc

showMap(mapPredict);