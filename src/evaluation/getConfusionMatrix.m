function [ cm ] = getConfusionMatrix( mapPredict, mapReference )
%GETCONFUSIONMATRIX Summary of this function goes here
%   Detailed explanation goes here
c = length(unique(mapReference)) - 1;

cm = zeros(c,c);

for al=1:c
    for pl=1:c
        cm(al,pl) = nnz(mapPredict(mapReference == al) == pl);
    end
end

end

