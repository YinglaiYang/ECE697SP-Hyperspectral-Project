function [ W ] = getRBF_AffinityMatrix( X, nystroemIDX, sigma )
%GETRBF_AFFINITYMATRIX Summary of this function goes here
%   Detailed explanation goes here

W = exp(-sqdist_mod(X, nystroemIDX) / (2*sigma^2));

end

