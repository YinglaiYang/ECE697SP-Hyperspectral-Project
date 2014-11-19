function [ meanValidationError, meanTrainingError ] = nFoldValidation( N, trainingData, trainingLabels, classificationFunc )
%NFOLDVALIDATION Summary of this function goes here
%   Input
%   -----
%   - N: how many partitions are used for cross-validation
%   - trainingData: data matrix, each row one sample, the columns are
%   different dimensions
%
%   Note
%   ----
%   In this implementation it is well possible that some samples will never
%   be part of the validation set, because I believe it is more important
%   that all validation sets have the same length. Also it is easier to
%   code.

N_samples = size(trainingData, 1);

L_validationSet = floor(N_samples/N);

validationError = zeros(1, N);
trainingError   = zeros(1, N);

% Cross-validation
for v=1:N    
    % Partitioning of data into training set and validatin set
    validationSetIDX = zeros(length(trainingData), 1);
    validationSetIDX(1+(v-1)*L_validationSet:v*L_validationSet) = 1;
    validationSetIDX = logical(validationSetIDX);
    
    trainingSetIDX = ~validationSetIDX;
    
    validationSetCV = trainingData(validationSetIDX,:);
    validationLabelsCV = trainingLabels(validationSetIDX,:);
    
    trainingSetCV = trainingData(trainingSetIDX,:);
    trainingLabelsCV = trainingLabels(trainingSetIDX,:);
    
    % Call of classifier function using training set onto the validation
    % set + using the training set on itself to calculate: validation error
    % and training error
    guessedValidationLabels = classificationFunc(trainingSetCV, trainingLabelsCV, validationSetCV);
    guessedTrainingLabels   = classificationFunc(trainingSetCV, trainingLabelsCV, trainingSetCV);
    
    guessedValidationLabels = guessedValidationLabels(:);
    guessedTrainingLabels   = guessedTrainingLabels(:);
    
    % Note the validation error 
    validationError(v) = sum(guessedValidationLabels ~= validationLabelsCV) / L_validationSet;
    trainingError(v)   = sum(guessedTrainingLabels ~= trainingLabelsCV) / (N_samples - L_validationSet);
end

% Calculate the average values over all validation errors
meanValidationError = mean(validationError);
meanTrainingError   = mean(trainingError);

end

