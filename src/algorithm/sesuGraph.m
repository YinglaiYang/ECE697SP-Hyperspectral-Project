function F_star = sesuGraph(Y, X, alpha, kernelFun, nystroemFraction)
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

m = round(nystroemFraction*n);

%% Nystroem method
NU = NystroemUniform(n, m);
NU_sampledIndices = NU.sampledIndices;

% calculate W_nm
W_nm = zeros(n, m);

for k=1:m
    disp(['k: ' num2str(k)]);
    W_nm(:,k) = getAffinitiesN(X, NU_sampledIndices(k), kernelFun);
end

W_mm = W_nm(NU_sampledIndices,:);

colsum_W_nm = sum(W_nm, 1);

d_n = ((colsum_W_nm * W_mm^-1) * W_nm.').';                                      clear W_mm;
d_m = d_n(NU_sampledIndices);

S_nm = spdiags(d_n.^-0.5, 0, n, n) * W_nm * sparse(diag(d_m.^-0.5));             clear W_nm;
S_mm = S_nm(NU_sampledIndices,:);

[V_mm, Lambda_mm] = eig(S_mm);

V_tilde = S_nm * V_mm;                                                     clear S_nm; clear V_mm;
Lambda_tilde = Lambda_mm^-1;                                               clear Lambda_mm;


end