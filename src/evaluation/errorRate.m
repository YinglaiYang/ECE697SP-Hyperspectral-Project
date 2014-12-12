function [ errorRate_result ] = errorRate( mapPredict, mapReference )
%ERRORRATE Gives the error rate of the predicted map compared to reference.
%Only the points where the reference map contains labels are compared.
%   Detailed explanation goes here

testIDX = mapReference > 0;

N_test = nnz(mapReference);

errorRate_result = nnz(mapPredict(testIDX) ~= mapReference(testIDX)) / N_test;

end

