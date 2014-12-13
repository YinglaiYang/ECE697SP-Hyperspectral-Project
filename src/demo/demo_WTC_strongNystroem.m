clc, clear, close all;
%% Runs about 115 seconds.

load hsi;
load mapTrain;

%% Parameters
alpha = 0.03;
sigma = 2.5e-2;

nystroemFraction = 0.01; % 1percent

windowHeight =100;
windowWidth = 200;

RSVD.k_fraction = 1.0;
RSVD.p = 20;
RSVD.q = 5;

tic;
%% Perform classification

mapPredict = windowedClassifierRSVD(hsi, mapTrain, alpha, sigma, nystroemFraction, RSVD, windowHeight, windowWidth);

% error rate
load mapTest;

testError = errorRate(mapPredict, mapTest);
confustionMatrix = getConfusionMatrix(mapPredict, mapTest);
[precision, recall] = precisionRecall(confustionMatrix);

toc

showMap(mapPredict);