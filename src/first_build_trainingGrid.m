clear, clc, close all;

load hsi;
load mapTrain;

Y = getLabelMatrixY(mapTrain);

[n, c] = size(Y);

q = 1;

grid = zeros(c*q,1);

for k=1:c
    tmp = find(mapTrain == k);
    
    grid((1+(k-1)*q):k*q) = tmp(1:q);
end

[~,~,d] = size(hsi);
X = reshape(hsi, n, d);
clear hsi;

runtime = tic;
F_star = sesuGraph_trainingGrid(Y, X, 0.1, @(x1,x2) RBF_kernel(x1, x2, 0.1), grid);
toc(runtime)

test