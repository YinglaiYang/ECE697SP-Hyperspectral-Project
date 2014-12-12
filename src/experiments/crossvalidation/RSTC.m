%% Test of classification algorithm with crossvalidation
% Tested algorithm: Randomized Sets, Tolerant Classifier (includes all
% samples, which means also the classless, black samples).
clear, clc, close all;

load hsi;
load mapTrain;
load mapTest; %Only location of test points are used; not the labels!!

c = 19;

Y = getLabelMatrixY(mapTrain, c);

[n, ~] = size(Y);

[~,~,d] = size(hsi);
X = reshape(hsi, n, d);

%% Specific parameters go here!
sectionSize = 10000;
testSetIDX = mapTest > 0;

%% Crossvalidation (5-fold)
[height, width] = size(mapTrain);

refIDX = find(mapTrain > 0); %Indices of the reference samples - use indirect linking later

cvIDX = crossvalind('Kfold', mapTrain(refIDX), 5); %Divide samples into five sets

%% Prepare crossvalidation with following parameters
parameters; %executes parameter scripts

best.alpha = 0;
best.sigma = 0;
best.cvErrorRate = 1;

errorOverSigma = zeros(length(sigmas), 1);

% Perform crossvalidation for different parameters
for a = 1:length(alphas)
    for s = 1:length(sigmas)
        alpha = alphas(a);
        sigma = sigmas(s);
        
        cvErrorRate = zeros(1, 5);
        
        for cv = 1:5   %5-fold crossvalidation
            cvTestIDX  = (cvIDX == cv);
            cvTrainIDX = (cvIDX ~= cv);
            
            cvMapTrain = zeros(size(mapTrain));
            cvMapTrain(refIDX(cvTrainIDX)) = mapTrain(refIDX(cvTrainIDX));
            
            cvMapTest = zeros(size(mapTrain));
            cvMapTest(refIDX(cvTestIDX)) = mapTrain(refIDX(cvTestIDX));
            
            % Do prediction
            predictionMap = randomizedSetsClassifier(hsi, cvMapTrain, alpha, sigma, nystroemFraction, RSVD, sectionSize);
            
            % Check error rate
            cvErrorRate(cv) = errorRate(predictionMap, cvMapTest);
        end
        
        avg_cvErrorRate = mean(cvErrorRate);
        
        if avg_cvErrorRate < best.cvErrorRate
            best.alpha = alpha;
            best.sigma = sigma;
            best.cvErrorRate = avg_cvErrorRate;
        end
        
        %% Have a look at the correlation to sigma
        errorOverSigma(s) = avg_cvErrorRate;
        
    end
    
    figure, semilogx(sigmas, errorOverSigma);
    title(sprintf('errors for \alpha = %d', alpha));
    xlabel('\sigma');
    ylabel('Cross validation error in percent');
    snapnow;
end


