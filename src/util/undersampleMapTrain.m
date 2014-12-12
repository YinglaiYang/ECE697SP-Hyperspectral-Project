function [ undersampledMapTrain ] = undersampleMapTrain( mapTrain, maxSamples )
%UNDERSAMPLEMAPTRAIN Summary of this function goes here
%   Detailed explanation goes here
c = length(unique(mapTrain)) - 1;

undersampledMapTrain = zeros(size(mapTrain));

for label=1:c
    idx = find(mapTrain == label);
    
    undersampledMapTrain(idx(1:maxSamples)) = label;
end

end

