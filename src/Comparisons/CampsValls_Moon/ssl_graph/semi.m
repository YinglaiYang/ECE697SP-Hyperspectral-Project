function  C = semi(X,Y, sigma,alpha)

% Input------
%   X --- the input matrix. Each row of the matix is the vector of a point. 
%   Y --- the label matrix.  For example, if there are 3 classes, then the
%           row vectors of Y are possibly [1 0  0], [0 1 0], [0, 0, 1], or [0, 0, 0].
%   sigma ---- the width of Gaussian if we choose a Gaussian to form an affinity
%                   matrix. If you have any prior affinity matrix, this
%                   paprameter is not necessary. 
%   alpha------the regularization parameter from 0 to 1. 
%
% Output------
%    C ---------- the final classification. The elements of C are numbers
%                 from 1 to k, where k is the number of classes. 
%

N = size(X, 1);

%================================================================
% Step 1: Affinity matrix
%================================================================
M = zeros(N, N); % norm matrix
for i = 1:N % compute the pairwise norm
    for j = (i+1):N
        M(i, j) = norm(X(i, :) - X(j, :)); 
        M(j, i) = M(i, j);
    end;
end;

% Use a Gaussian to form an affinity matrix
K = exp(-M.^2/(2*sigma^2));  

% zero diag. very very important! 
%pcolor(K),shading interp, colorbar,pause

K = K - eye(N); 

%figure,pcolor(K),shading interp, colorbar,pause
%================================================================
% Step 2: Symmetrical normalization
%================================================================
D = diag(1./sqrt(sum(K))); % the inverse of the square root of the degree matrix

S = D*K*D; %normalize the affinity matrix

% D = diag(1./sum(K));
% S = D*K; 

%================================================================
% Step 3(a): Compute the classification function
%================================================================
F = inv(eye(N) - alpha * S) * Y; % classification function F = (I - \alpha S)^{-1}Y

%================================================================
% Step 3(b):  Classifying the data
%================================================================
[dummy, C] = max(F, [], 2); %simply checking which of elements is largest in each row
