clear, clc, close all;

load hsi;
load mapTrain;

Y = getLabelMatrixY(mapTrain);

[n, c] = size(Y);

[~,~,d] = size(hsi);
X = reshape(hsi, n, d);
clear hsi;

runtime = tic;
F_star = sesuGraph(Y, X, 0.1, @(x1,x2) RBF_kernel(x1, x2, 0.7), 5e-4, 1);
toc(runtime)