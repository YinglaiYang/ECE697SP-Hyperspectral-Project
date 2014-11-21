function [ affinity ] = RBF_kernel( x1, x2, sigma )
%RBF_KERNEL Summary of this function goes here
%   Detailed explanation goes here

lastDim = ndims(x1);

affinity = exp( -sum((x1 - x2).^2, lastDim) / (2*sigma^2) );

end

