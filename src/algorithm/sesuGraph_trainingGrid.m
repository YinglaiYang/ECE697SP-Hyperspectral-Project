function F_star = sesuGraph_trainingGrid(Y, X, alpha, kernelFun, grid)
%SESUGRAPH_01 Semi-supervised graph-based image classification for our
%hyperspectral project. 
%
% First version (prototype). Dumbed down: It only uses spectral features
% and a simple Nyström algorithm with uniformly distributed sampling.
%
% Following variables are used to show matrix sizes: 
% # n - number of pixels in the image
% # d - number of dimensions of the features. Here, length of spectral
%       data.
% # c - number of classes. Here: 19 classes (different minerals).
%
% Input
% =====
% # Y - [n,c]-matrix | The "seed" of the algorithm. Each row shows the
%       training label of the corresponding pixel in the image. If the
%       pixel is part of the training set, the column corresponding to the
%       training label will hold a '1', while the rest are '0'. If the 
%       pixel is not part of the training set, all columns will hold '0'.
%
% # X - [n,d]-matrix | The matrix holding all the features (spectral data,
%       here) for all the pixels. This will be used to calculate the
%       similarties (The similarity matrix itself will not be calculated;
%       it is simply too big. It is a [n,n]-matrix, which means almost
%       3.6e11 entries! It **will** block your memory!)
%
% # kernelFun - function handle: double = @affinityFun([double] x1, [double] x2) |
%               A function handle that returns the affinity between two
%               feature sets. Takes two vectors.
%
% Output
% ======
% # F_star - [n,c]-matrix | The result of the algorithm, a matrix where each row is
%            a pixel in the image. One column per row contains a '1' to
%            denote that the pixel of that row was classified as the class
%            of that column.

%% Parameters etc.
[n, c] = size(Y);

m = length(grid); %0 is not a class
p = 10;

%% Nystroem method
NU_sampledIndices = grid;

% calculate W_nm
W_nm = zeros(n, m);

for k=1:m
    disp(['k: ' num2str(k)]);
    W_nm(:,k) = getAffinities2(X, NU_sampledIndices(k), kernelFun);
end

abnormalityCheck(W_nm);

clear X;

W_mm = W_nm(NU_sampledIndices,:);
abnormalityCheck(W_mm);
symmetryCheck(W_mm);

% save('tmp_Wnm.mat', 'W_nm');

% memory;

[V_W_mm, Lambda_W_mm] = eigs(W_mm, p);

colsum_V_W_tilde = sum(sqrt(m/n) * W_nm * (V_W_mm * Lambda_W_mm^-1), 1);
abnormalityCheck(colsum_V_W_tilde);

d_n = ((colsum_V_W_tilde * (n/m) * Lambda_W_mm) * (sqrt(m/n) * W_nm * (V_W_mm * Lambda_W_mm^-1)).').';  
d_n = d_n - 1;

abnormalityCheck(d_n);
fprintf('d_n has %d negative elements.\n', numel(find(d_n < 0)) );

d_m = d_n(NU_sampledIndices);
abnormalityCheck(d_m);

S_nm = spdiags(1./sqrt(d_n(:)), 0, n, n) * ... 
            ( W_nm * spdiags(1./sqrt(d_m(:)), 0, m, m) );                     clear W_nm;

% S_nm = zeros(size(W_nm));
% 
% for i=1:size(W_nm, 1)
%     for j=1:size(W_nm, 2)
%         S_nm(i,j) = W_nm(i,j) / (sqrt(d_n(i)) * sqrt(d_m(j)));
%     end
% end

clear W_nm;

abnormalityCheck(S_nm);

% memory;

%Disabled section - this version makes S_mm unsymmetric
% S_mm = spdiags(d_m(:).^-0.5, 0, m, m) * ... 
%             ( W_mm * spdiags(d_m(:).^-0.5, 0, m, m) );                     clear W_mm;

S_mm = zeros(size(W_mm));

for i=1:size(W_mm, 1)
    for j=1:size(W_mm, 2)
        S_mm(i,j) = W_mm(i,j) / (sqrt(d_m(i)) * sqrt(d_m(j)));
    end
end

                                                                           clear W_mm;

abnormalityCheck(S_mm);
symmetryCheck(S_mm);

[V_mp, Lambda_pp] = eigs(S_mm, p);                                         clear S_mm;

V_tilde = sqrt(n/m) * S_nm * (V_mp * Lambda_pp^-1);                        clear S_nm; clear V_mp;
Lambda_tilde = (n/m) * Lambda_pp;                                          clear Lambda_pp;

% memory;

A_inv = getAinv(V_tilde, Lambda_tilde, alpha);

evaluation1 = ( Lambda_tilde * (V_tilde.' * A_inv * V_tilde) ...
                  - spdiags(alpha^-1 * ones(p,1), 0, p, p) )^-1;
              
evaluation2 = A_inv * V_tilde * ( evaluation1 * ...
                              (Lambda_tilde * (V_tilde.' * A_inv * Y)) );  clear evaluation1;
                          
F_star = (1-alpha) * (A_inv * Y - evaluation2);                       

end