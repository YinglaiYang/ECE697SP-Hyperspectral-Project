clear, clc, close all;

load hsi;
load mapTrain;

Y = getLabelMatrixY(mapTrain);

runtime = tic;
sesuGraph(Y, hsi, @(x1,x2) RBF_kernel(x1, x2, 0.01), 0.001);
toc(runtime)