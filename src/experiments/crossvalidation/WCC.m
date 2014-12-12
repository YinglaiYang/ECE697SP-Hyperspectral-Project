%% Test of classification algorithm with crossvalidation
clear, clc, close all;

load hsi;
load mapTrain;

c = 19;

Y = getLabelMatrixY(mapTrain, c);

[n, ~] = size(Y);

[~,~,d] = size(hsi);
X = reshape(hsi, n, d);

%% Specific parameters go here!
windowHeight = 100;
windowWidth = 100;

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
            try
                testIDX  = (cvIDX == cv);
                trainIDX = (cvIDX ~= cv);

                cvMapTrain = zeros(size(mapTrain));
                cvMapTrain(refIDX(trainIDX)) = mapTrain(refIDX(trainIDX));

                cvMapTest = zeros(size(mapTrain));
                cvMapTest(refIDX(testIDX)) = mapTrain(refIDX(testIDX));
                
                testSetIDX = cvMapTest > 0;

                % Do prediction
                predictionMap = windowedClassistClassifier(hsi, cvMapTrain, testSetIDX, alpha, sigma, nystroemFraction, RSVD, windowHeight, windowWidth);

                % Check error rate
                cvErrorRate(cv) = errorRate(predictionMap, cvMapTest);
            catch
                cvErrorRate(cv) = 1;
                disp('######### Something went wrong #########');
            end
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
    title(sprintf('errors for \\alpha = %d', alpha));
    xlabel('\sigma');
    ylabel('Cross validation error in percent');
    snapnow;
end


