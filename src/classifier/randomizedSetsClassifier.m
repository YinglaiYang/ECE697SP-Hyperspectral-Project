function [ predictedMap ] = randomizedSetsClassifier( hsi, mapTrain, alpha, sigma, nystroemFraction, RSVD, sectionSize )
%RANDOMIZEDSECTIONSCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
%% Parameters
[imgHeight, imgWidth, d] = size(hsi);
labels_including_0 = unique(mapTrain);
c = length(labels_including_0) - 1; %How many "real" class labels are there

N_samples = imgHeight*imgWidth;

X = reshape(hsi, [N_samples, d]);

N_sections = ceil(N_samples / sectionSize);

predictedMap = zeros(N_samples, 1);

%% Classification function
get_f_star = @(Y, X) sesuGraph_RSVD(Y, X, alpha, sigma, nystroemFraction, RSVD);

%% Create the sectioning
sections = crossvalind('Kfold', size(X,1), N_sections);

%% Perform classification for each section
for s=1:N_sections
    sectionIDX = (sections == s);
    
    outerTrainingIDX = (mapTrain(:) > 0) & ~sectionIDX;
    
    newX = [X(sectionIDX,:); X(outerTrainingIDX,:)];
    newY = getLabelMatrixY(mapTrain(sectionIDX | outerTrainingIDX), c);
    
    f_star = get_f_star(newY, newX);
    predictedLabels = predictedLabelsFromFstar(f_star);
    
    predictedMap(sectionIDX) = predictedLabels(1:nnz(sectionIDX));
end

predictedMap = reshape(predictedMap, [imgHeight, imgWidth]);

end

