function mapPredict = predictionMapFromFstar( F_star, width, height )
%PREDICTIONMAPFROMFSTAR Summary of this function goes here
%   Detailed explanation goes here

[~, predictedLabels] = max(F_star, [], 2); %position of maximum value in each row is the predicted class label

mapPredict = reshape(predictedLabels, height, width);

end

