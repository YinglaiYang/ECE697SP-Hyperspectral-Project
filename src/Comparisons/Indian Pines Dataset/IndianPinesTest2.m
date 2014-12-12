clc, clear, close all;

load ipMapTrain;
load ipMapTest;
load subip_hsi;

[n1, n2, d] = size(subip_hsi);
c = length(unique(ipMapTrain)) - 1; %should be 4

ipMapTrain2 = zeros(size(ipMapTrain));

N_labels = 3;
for label=1:c
    pos = find(ipMapTrain==label);
    chosen_pos = randsample(length(pos), N_labels);
    
    ipMapTrain2(pos(chosen_pos)) = label;
end

X = reshape(subip_hsi, [n1*n2, d]);
Y = getLabelMatrixY(ipMapTrain2, c);

%% Classification parameters
nystroemFraction = 0.2; % 1percent

RSVD.k_fraction = 0.7;
RSVD.p = 10;
RSVD.q = 3;

alpha = 0.01;
sigma = 1.15e3;

%% Test samples individually
N_testSamples = nnz(ipMapTest);
testSamplesPos = find(ipMapTest > 0);

trainIDX = ipMapTrain2 > 0;

mapPredict = zeros(size(ipMapTrain2));

newX = [X(testSamplesPos,:); X(trainIDX(:),:)];
newY = [Y(testSamplesPos,:); Y(trainIDX(:),:)];

%% Perform classification
get_f_star = @(Y, X) sesuGraph_preciseD(Y, X, alpha, sigma, nystroemFraction, RSVD);
f_star = get_f_star(newY, newX);
tmp = predictedLabelsFromFstar(f_star);
mapPredictVec = tmp(1:length(testSamplesPos));

mapPredict(testSamplesPos) = mapPredictVec;

%% Error rate
errorRate = errorRate(mapPredict, ipMapTest);
showMap(mapPredict);