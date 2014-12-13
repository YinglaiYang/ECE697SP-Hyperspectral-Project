%% Careful! This script needs about 90 minutes to run!

clc, clear, close all;

load hsi;
load mapTrain;

%% Parameters
alpha = 0.1;
sigma = 0.025;

nystroemFraction = 1e-1; % 1percent

windowHeight = 108;
windowWidth = 103;

RSVD.k_fraction = 0.7;
RSVD.p = 10;
RSVD.q = 3;

tic;
%% Perform classification
mapPredict = windowedClassifierRSVD(hsi, mapTrain, alpha, sigma, nystroemFraction, RSVD, windowHeight, windowWidth);

% error rate
load mapTest;

testError = errorRate(mapPredict, mapTest);
confusionMatrix = getConfusionMatrix(mapPredict, mapTest);
[precision, recall] = precisionRecall(confusionMatrix);

toc

showMap(mapPredict);