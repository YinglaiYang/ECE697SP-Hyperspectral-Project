function [ predictedMap ] = windowedClassifierRSVD( hsi, mapTrain, alpha, sigma, nystroemFraction, RSVD, sectionWidth, sectionHeight )
%SECTIONEDCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here

%% Parameters
[imgHeight, imgWidth, d] = size(hsi);
labels_including_0 = unique(mapTrain);
c = length(labels_including_0) - 1; %How many "real" class labels are there
predictedMap = zeros(imgHeight, imgWidth);

get_f_star = @(Y, X) sesuGraph_RSVD(Y, X, alpha, sigma, nystroemFraction, RSVD);

%% 
% Go through each section in the image and perform the classification
% algorithm on these sections using the complete training set.
for i = 1:ceil(imgHeight/sectionHeight)
    for j = 1:ceil(imgWidth/sectionWidth)
        if(i < ceil(imgHeight/sectionHeight) &&  j < ceil(imgWidth/sectionWidth))
            window = [(i-1)*sectionHeight+1, (j-1)*sectionWidth+1, i*sectionHeight, j*sectionWidth];
            disp(['window: ' num2str(window)]);   
        elseif( i < ceil(imgHeight/sectionHeight) && j == ceil(imgWidth/sectionWidth))
            window = [(i-1)*sectionHeight+1, imgWidth-sectionWidth+1, i*sectionHeight, imgWidth];
            disp(['window: ' num2str(window)]);
        elseif(i == ceil(imgHeight/sectionHeight) && j < ceil(imgWidth/sectionWidth))
            window = [imgHeight-sectionHeight+1, (j-1)*sectionWidth+1, imgHeight, j*sectionWidth];
            disp(['window: ' num2str(window)]);
        elseif(i == ceil(imgHeight/sectionHeight) &&  j == ceil(imgWidth/sectionWidth))
            window = [imgHeight-sectionHeight+1, imgWidth-sectionWidth+1, imgHeight, imgWidth];          
            disp(['window: ' num2str(window)]);
        end
        
        
        
        [~, newX, newY] = getSubwindowData(mapTrain, hsi, window);
        f_star = get_f_star(newY, newX);
        predictedMap(window(1):window(3), window(2):window(4)) = ...
                predictionMapFromFstar(f_star(1:sectionWidth*sectionHeight,:), sectionWidth, sectionHeight);
    end
end

end

