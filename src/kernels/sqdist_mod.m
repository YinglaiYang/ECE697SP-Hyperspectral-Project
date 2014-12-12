function [ dist2 ] = sqdist_mod( X, nystroemIDX )
%SQDIST_MOD Fast method to calculate the squared distance for use in
%kernel.
%
%   This is slight modification of the function `sqdist.m` which was packed
%   together with the Improved_Nystrom_Method. Credits go to `Roland
%   Bunschoten` (s. original file).
%
% Input
% =====
% # X - [nxd] data matrix. `n` is the number of samples and `d` the length 
%       of the feature vector. Consists of stacked row vectors, where each 
%       row vector stands for the features of one sample point.
%
% # nystroemIDX - Length m vector. `m` stands for the number of Nystroem
%                 samples taken. Each element holds the index of the
%                 corresponding Nystroem column/row of the whole matrix.
%
% Output
% ======
% # dist2 - [nxm] distance matrix. Squared distance. Squared L2 norm of
%           difference between vectors.

[n, ~] = size(X);
m = length(nystroemIDX);

% X_part = X(nystroemIDX,:);

a2 = sum(X.*X, 2);
% b2 = sum(X_part.*X_part, 2);
b2 = sum(X(nystroemIDX,:) .* X(nystroemIDX,:), 2);

AB = X * X(nystroemIDX,:).';

dist2 = abs(repmat(a2(:), [1 m]) + repmat(b2(:).', [n 1]) - 2*AB);

end

