function [ Ainv ] = getAinv( V, D, alpha )
%GETAINV
%   Calculates A_inv for the Woodbury evaluation formula with the modified
%   Nyström-method.

[n1, ~] = size(V);

diagonal_vec = zeros(n1, 1);

% Diagonal = spalloc(n1, n1, n1); %diagonal matrix! nonzero elements only on diagonal

for i=1:n1
    diagonal_vec(i) = V(i,:) * D * V(i,:).';
end

diagonal_vec = (1 + alpha*diagonal_vec).^-1;

Ainv = spdiags(diagonal_vec(:), 0, n1, n1);

end

