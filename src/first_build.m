clear, clc, close all;

load hsi;
load mapTrain;

Y = getLabelMatrixY(mapTrain);

[n, c] = size(Y);

[~,~,d] = size(hsi);
X = reshape(hsi, n, d);
clear hsi;

runtime = tic;
sesuGraph(Y, X, 0.5, @(x1,x2) RBF_kernel(x1, x2, 0.01), 0.0001);
toc(runtime)