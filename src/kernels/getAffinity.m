function [ affinity ] = getAffinity( X, i, j, kernelFun )
%GETAFFINITY Summary of this function goes here
%   Detailed explanation goes here
if i==j
    affinity = 0;
else
    affinity = kernelFun(X(i,:), X(j,:));
end

end

