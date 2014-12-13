clc, clear, close all;

load hsi;
load mapTrain;

%% Parameters
alpha = 0.03;
sigma = 2.5e-2;

nystroemFraction = 0.001; % 1percent

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
cm = getConfusionMatrix(mapPredict, mapTest);
[precision, recall] = precisionRecall(cm);

toc