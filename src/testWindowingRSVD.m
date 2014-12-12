clc, clear, close all;

load hsi;
load mapTrain;

%% Parameters
alpha = 0.1;
sigma = 2.5e-2;

nystroemFraction = 1e-3; % 1percent

windowHeight = 100;
windowWidth = 100;

RSVD.k_fraction = 0.7;
RSVD.p = 10;
RSVD.q = 3;

tic;
%% Perform classification
mapPredict = windowedClassifierRSVD(hsi, mapTrain, alpha, sigma, nystroemFraction, RSVD, windowHeight, windowWidth);

% error rate
load mapTest;

testError = errorRate(mapPredict, mapTest);
cm = getConfusionMatrix(mapPredict, mapTest);
[precision, recall] = precisionRecall(cm);

toc