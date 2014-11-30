function [ errorRate ] = errorRate( mapPredict, mapReference )
%ERRORRATE Summary of this function goes here
%   Detailed explanation goes here

testIDX = mapReference > 0;

N_test = nnz(mapReference);

errorRate = nnz(mapPredict(testIDX) ~= mapReference(testIDX)) / N_test;

end

