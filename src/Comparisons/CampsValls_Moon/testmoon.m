load x.mat;
load y.mat;
load z.mat;
load c.mat;

z2 = Z;
z2(find(z2==-1)) =2;
%% Parameters
alpha = 0.99;
sigma = 0.1;

nystroemFraction = 0.1; % 1percent

tic;
%% Perform classification
get_f_star = @(Y, X) sesuGraph_RSVD(Y, X, alpha, sigma, nystroemFraction);
Y = getLabelMatrixY(z2, 2);
f_star = get_f_star(Y, X);
predict = predictionMapFromFstar(f_star,1,200);
%mapPredict = windowedClassifierRSVD(X, z2, alpha, sigma, nystroemFraction, windowHeight, windowWidth);
