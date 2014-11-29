function [ affinities ] = getAffinities2( X, j, kernelFun )
%GETAFFINITIES Summary of this function goes here
%   Detailed explanation goes here
[n,~] = size(X);

affinities = kernelFun(X, repmat(X(j,:), n, 1));

% affinities = affinities(:); %not necessary?


end

