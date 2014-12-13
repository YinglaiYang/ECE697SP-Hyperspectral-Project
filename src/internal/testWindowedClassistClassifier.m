clc, clear, close all;

load hsi;
load mapTrain;
load mapTest;
load trainingSamplesPerClass;

%% Parameters
alpha = 0.1;
sigma = 2.5e-2;

nystroemFraction = 1e-3; % 1percent

sectionHeight = 100;
sectionWidth  = 100;

RSVD.k_fraction = 0.7;
RSVD.p = 10;
RSVD.q = 3;

testIDX = mapTest > 0;

tic;
%% Perform classification
predictedLabels = windowedClassistClassifier(hsi, mapTrain, testIDX, alpha, sigma, nystroemFraction, RSVD, sectionWidth, sectionHeight);

mapPredict = reshape(predictedLabels, size(mapTrain));

%% Error rate
testError = errorRate(mapPredict, mapTest);
cm = getConfusionMatrix(mapPredict, mapTest);
[precision, recall] = precisionRecall(cm);

toc