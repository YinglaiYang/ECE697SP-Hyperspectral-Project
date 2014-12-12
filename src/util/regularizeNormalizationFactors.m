function [ regularized_d_vector ] = regularizeNormalizationFactors( d_vector, factor )
%REGULARIZENORMALIZATIONFACTOR Summary of this function goes here
%   Detailed explanation goes here

idx_zero = (d_vector == 0);

min_nonzero_value = min(d_vector(~idx_zero));

regularization_value = min_nonzero_value * factor; %assign the zero values a smaller value

regularized_d_vector = d_vector;
regularized_d_vector(idx_zero) = regularization_value;

end

