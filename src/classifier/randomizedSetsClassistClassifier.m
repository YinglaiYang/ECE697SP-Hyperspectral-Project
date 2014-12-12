function [ predictedLabels, F_star ] = randomizedSetsClassistClassifier( hsi, mapTrain, testSamplesIDX, alpha, sigma, nystroemFraction, RSVD, sectionSize )
%CLASSISTCLASSIFIER CLassification only with samples provided by training
%set (with labels) and test set (without labels).
%   Idea: In the given image `hsi`, "black" samples could be very bad for
%   the classification. This "classist" algorithm makes sure to exclude the
%   "black" samples by only using samples that are "properly colored".

%% Parameters
[imgHeight, imgWidth, d] = size(hsi);
labels_including_0 = unique(mapTrain);
c = length(labels_including_0) - 1; %How many "real" class labels are there

N_samples = imgHeight*imgWidth;

N_testSamples = nnz(testSamplesIDX);

N_trustySamples = nnz(mapTrain) + N_testSamples;

X_full = reshape(hsi, [N_samples, d]);

trustyIDX = (mapTrain(:) > 0) | testSamplesIDX(:);

X_trusty = X_full(trustyIDX);



N_sections = ceil(N_trustySamples / sectionSize);

predictedLabels = zeros(N_samples, 1);
F_star = zeros(N_samples, c);

%% classification function
get_f_star = @(Y, X) sesuGraph_RSVD(Y, X, alpha, sigma, nystroemFraction, RSVD);

%% Create the sectioning
sections = crossvalind('Kfold', size(X_trusty,1), N_sections);

%% Perform classification for each section
for s=1:N_sections
    trustySectionIDX = (sections == s);
    fullSectionIDX = trustyToFull_IDX(trustyIDX, trustySectionIDX);
    
    trustyOuterTrainingIDX = (mapTrain(trustyIDX) > 0) & ~trustySectionIDX(:);
    fullOuterTrainingIDX = trustyToFull_IDX(trustyIDX, trustyOuterTrainingIDX);
    
    newX = [X_trusty(trustySectionIDX,:); X_trusty(trustyOuterTrainingIDX,:)];
    
    Y_section       = getLabelMatrixY(mapTrain(fullSectionIDX), c);
    Y_outerTraining = getLabelMatrixY(mapTrain(fullOuterTrainingIDX), c);
    
    newY = [Y_section; Y_outerTraining];
    
    fprintf('~~~~~~~~~~~~ %d training samples used for section %d ~~~~~~~~~~~~\n', nnz(Y_section)+nnz(Y_outerTraining), s);
    
    f_star = get_f_star(newY, newX);
    
    F_star(fullSectionIDX,:) = f_star(1:nnz(trustySectionIDX),:);
    
    predictedSectionLabels = predictedLabelsFromFstar(f_star);
    
    fullSectionIDX = trustyToFull_IDX(trustyIDX, trustySectionIDX);
    
    predictedLabels(fullSectionIDX) = predictedSectionLabels(1:nnz(trustySectionIDX));
end

end

function fullIDX = trustyToFull_IDX(trustyIDX, trustySubIDX)
fullTrustyPoints = find(trustyIDX);

fullIDX = false(size(trustyIDX));

fullIDX(fullTrustyPoints(trustySubIDX)) = true;
end

