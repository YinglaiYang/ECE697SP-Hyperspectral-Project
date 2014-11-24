function F_star = sesuGraph(Y, X, alpha, kernelFun, nystroemFraction, pFraction)
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

m = round(nystroemFraction * n);
p = ceil(pFraction * m);

%% Nystroem method
NU = NystroemUniform(n, m);
NU_sampledIndices = NU.sampledIndices;

l = max(NU_sampledIndices);

% calculate W_nm
W_nm = zeros(n, m);

for k=1:m
    disp(['k: ' num2str(k)]);
    W_nm(:,k) = getAffinities(X, NU_sampledIndices(k), kernelFun);
end

W_mm = W_nm(NU_sampledIndices,:);
abnormalityCheck(W_mm);

colsum_W_nm = sum(W_nm, 1);
abnormalityCheck(colsum_W_nm);

d_n = ((colsum_W_nm * W_mm^-1) * W_nm.').';                                clear W_mm;
abnormalityCheck(d_n);
d_m = d_n(NU_sampledIndices);
abnormalityCheck(d_m);

S_nm = spdiags(d_n(:).^-0.5, 0, n, n) * ... 
            ( W_nm * spdiags(d_m(:).^-0.5, 0, m, m) );                     clear W_nm;
% S_nm = W_nm;                                                               clear W_nm;

abnormalityCheck(S_nm);

S_mm = S_nm(NU_sampledIndices,:);
abnormalityCheck(S_mm);

[V_mp, Lambda_pp] = eigs(S_mm, p);                                         clear S_mm;

V_tilde = sqrt(m/n) * S_nm * (V_mp * Lambda_pp^-1);                        clear S_nm; clear V_mp;
Lambda_tilde = (n/m) * Lambda_pp;                                               clear Lambda_pp;

evaluation1 = ( Lambda_tilde * (V_tilde.' * V_tilde) ...
                  - spdiags(alpha^-1 * ones(p,1), 0, p, p) )^-1;
              
evaluation2 = V_tilde * ( evaluation1 * ...
                              (Lambda_tilde * (V_tilde.' * Y)) );          clear evaluation1;
                          
F_star = (1-alpha) * (Y - evaluation2);                       

end