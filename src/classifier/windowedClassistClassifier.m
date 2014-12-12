function [ predictedLabels, F_star ] = windowedClassistClassifier( hsi, mapTrain, testSamplesIDX, alpha, sigma, nystroemFraction, RSVD, sectionWidth, sectionHeight )
%WINDOWEDCLASSISTCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here

%% Parameters
[imgHeight, imgWidth, d] = size(hsi);
labels_including_0 = unique(mapTrain);
c = length(labels_including_0) - 1; %How many "real" class labels are there

N_samples = imgHeight*imgWidth;

N_testSamples = nnz(testSamplesIDX);

N_trustySamples = nnz(mapTrain) + N_testSamples;

X_full = reshape(hsi, [N_samples, d]);

trustyIDX = (mapTrain(:) > 0) | testSamplesIDX(:);

predictedLabels = zeros(N_samples, 1);
F_star = zeros(N_samples, c);

%% Classification function
get_f_star = @(Y, X) sesuGraph_RSVD(Y, X, alpha, sigma, nystroemFraction, RSVD);

%% Go through windowing
%% 
% Go through each section in the image and perform the classification
% algorithm on these sections using the complete training set.
for i = 1:ceil(imgHeight/sectionHeight)
    for j = 1:ceil(imgWidth/sectionWidth)
        if(i < ceil(imgHeight/sectionHeight) &&  j < ceil(imgWidth/sectionWidth))
            window = [(i-1)*sectionHeight+1, (j-1)*sectionWidth+1, i*sectionHeight, j*sectionWidth];
        elseif( i < ceil(imgHeight/sectionHeight) && j == ceil(imgWidth/sectionWidth))
            window = [(i-1)*sectionHeight+1, imgWidth-sectionWidth+1, i*sectionHeight, imgWidth];
        elseif(i == ceil(imgHeight/sectionHeight) && j < ceil(imgWidth/sectionWidth))
            window = [imgHeight-sectionHeight+1, (j-1)*sectionWidth+1, imgHeight, j*sectionWidth];
        elseif(i == ceil(imgHeight/sectionHeight) &&  j == ceil(imgWidth/sectionWidth))
            window = [imgHeight-sectionHeight+1, imgWidth-sectionWidth+1, imgHeight, imgWidth];          
        end
        
        disp(['window: ' num2str(window)]);
        
        windowSectionIDX = false(size(mapTrain));
        windowSectionIDX(window(1):window(3), window(2):window(4)) = true;
        windowSectionIDX = windowSectionIDX(:);
        
        newX_inner = X_full(trustyIDX & windowSectionIDX,:);
        newX_outer = X_full(mapTrain(:) > 0 & ~windowSectionIDX,:);
        newX = [newX_inner; newX_outer];
        
        newY_inner = getLabelMatrixY(mapTrain(trustyIDX & windowSectionIDX), c);
        newY_outer = getLabelMatrixY(mapTrain(mapTrain(:) > 0 & ~windowSectionIDX), c);
        newY = [newY_inner; newY_outer];
        
        
        N_innerSamples = size(newY_inner, 1);
        
        f_star = get_f_star(newY, newX);
        predictedLabels(trustyIDX &windowSectionIDX) = ...
                predictedLabelsFromFstar(f_star(1:N_innerSamples,:));
            
        F_star(trustyIDX & windowSectionIDX,:) = f_star(1:N_innerSamples,:);
    end
end

end

