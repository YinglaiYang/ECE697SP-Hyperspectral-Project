function [ precision, recall ] = precisionRecall( cm )
%PRECISIONRECALL Summary of this function goes here
%   Detailed explanation goes here
c = size(cm,1);

precision = zeros(c,1);
recall    = zeros(c,1);

for label=1:c
    precision(label) = cm(label,label) / sum(cm(:,label));
    if cm(label,label) == 0
        precision(label) = 0;
    end
    
    recall(label)    = cm(label,label) / sum(cm(label,:));
end

end

