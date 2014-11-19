function F_star = sesuGraph_01(Y, X, affinityFun)
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
% # affinityFun - function handle: double = @affinityFun([double] x1, [double] x2) |
%                 A function handle that returns the affinity between two
%                 feature sets. Takes two vectors.
%
% Output
% ======
% # F_star - [n,c]-matrix | The result of the algorithm, a matrix where each row is
%            a pixel in the image. One column per row contains a '1' to
%            denote that the pixel of that row was classified as the class
%            of that column.


end