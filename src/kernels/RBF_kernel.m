function [ affinity ] = RBF_kernel( x1, x2, sigma )
%RBF_KERNEL Summary of this function goes here
%   Detailed explanation goes here

affinity = exp( -norm(x1 - x2)^2 / (2*sigma^2) );

end

