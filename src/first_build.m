clear, clc, close all;

load hsi;
load mapTrain;

Y = getLabelMatrixY(mapTrain);

[n, c] = size(Y);

[~,~,d] = size(hsi);
X = reshape(hsi, n, d);
clear hsi;

runtime = tic;
F_star = sesuGraph_RSVD(Y, X, 0.1, 0.01, 1e-4);
toc(runtime)

