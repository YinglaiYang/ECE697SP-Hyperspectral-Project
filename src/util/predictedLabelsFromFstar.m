function [ predictedLabels ] = predictedLabelsFromFstar( F_star )
%PREDICTIONLABELSFROMFSTAR Summary of this function goes here
%   Detailed explanation goes here

[maxF, predictedLabels] = max(F_star, [], 2); %position of maximum value in each row is the predicted class label

predictedLabels(maxF == 0) = 0;
end

