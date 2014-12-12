function [ occ_vec ] = occurences( matrix, vals )
%OCCURENCES Summary of this function goes here
%   Detailed explanation goes here
occ_vec = zeros(1,length(vals));

for v=1:length(vals)
    occ_vec(v) = nnz(matrix == vals(v));
end

end

