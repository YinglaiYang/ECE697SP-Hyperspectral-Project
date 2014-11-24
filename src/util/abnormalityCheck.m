function [ infFlag, nanFlag ] = abnormalityCheck( variable )
%ABNORMALITYCHECK Summary of this function goes here
%   Detailed explanation goes here

infFlag = any(isinf(variable(:)));

nanFlag = any(isnan(variable(:)));

fprintf('%s - Inf: %d, NaN: %d\n', inputname(1), infFlag, nanFlag);
end

