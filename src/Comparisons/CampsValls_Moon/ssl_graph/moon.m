function [X, Y] = moon

% Two moons data

%==========================================================================
% Some parameters
%==========================================================================
N1 = 100; N2 = 100; N = N1+N2;
c1 = [0  -0.2]; c2 = [0.9 0.25];
r1 = 1.3; r2 = 1;
theta1 = 0.12; theta2 = 0.12;
rand('state', 1); randn('state', 1);

%==========================================================================
% Generate the dataset
%==========================================================================
ang1 = rand(N1, 1) * pi;
dis1 = randn(N1, 1) * theta1 + r1;
rel1 = [dis1 .* cos(ang1) dis1 .* sin(ang1)];
cen1 = repmat(c1, N1, 1);
abs1 = cen1 + rel1;

ang2 = rand(N2, 1) * pi + pi;
dis2 = randn(N2, 1) * theta2 + r1;
rel2 = [dis2 .* cos(ang2) dis2 .* sin(ang2)];
cen2 = repmat(c2, N2, 1);
abs2 = cen2 + rel2;

X = [abs1; abs2];

%==========================================================================
% One labeled point for each class
%==========================================================================
Y = zeros(N,2);
[dummy label_1] = min(X(1:N1, 1));
[dummy label_2] = max(X(N1+1:N, 1));
Y(label_1, :) = [1 0];
label_2 = label_2 + N1; 
Y(label_2, :) = [0 1]; 

