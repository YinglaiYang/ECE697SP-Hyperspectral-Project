function [ norm_F_star ] = normalizeF_star( F_star, trainingsamplesPerClass )
%NORMALIZEF_STAR Summary of this function goes here
%   Detailed explanation goes here

normFactor = sum(trainingsamplesPerClass)./trainingsamplesPerClass;
normFactor = normFactor(:).'; %force row vector

norm_F_star = F_star .* repmat(normFactor, size(F_star, 1), 1);

end

