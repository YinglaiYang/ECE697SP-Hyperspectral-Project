clc, clear, close all;

load ipMapTrain;
load ipMapTest;
load subip_hsi;

[n1, n2, d] = size(subip_hsi);
c = length(unique(ipMapTrain)) - 1; %should be 4

X = reshape(subip_hsi, [n1*n2, d]);
Y = getLabelMatrixY(ipMapTrain, c);

%% Classification parameters
nystroemFraction = nystroemFractions(nf); % 1percent

RSVD.k_fraction = 1.0;
RSVD.p = 10;
RSVD.q = 3;

alpha = 0.1;
sigma = 1e2;

%% Perform classification
get_f_star = @(Y, X) sesuGraph_preciseD(Y, X, alpha, sigma, nystroemFraction, RSVD);
f_star = get_f_star(Y, X);
mapPredict = predictionMapFromFstar(f_star, size(ipMapTrain,2), size(ipMapTrain,1));

%% Error rate
errorRate = errorRate(mapPredict, ipMapTest);
showMap(mapPredict);