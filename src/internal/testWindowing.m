clc, clear, close all;

load hsi;
load mapTrain;

%% Parameters
alpha = 0.1;
sigma = 2.5e-2;

nystroemFraction = 1e-3; % 1percent

windowHeight = 100;
windowWidth = 100;

tic;
%% Perform classification
mapPredict = windowedClassifier(hsi, mapTrain, alpha, sigma, nystroemFraction, windowWidth, windowHeight);

%% error rate
load mapTest;

testError = errorRate(mapPredict, mapTest);
cm = getConfusionMatrix(mapPredict, mapTest);
[precision, recall] = precisionRecall(cm);

toc