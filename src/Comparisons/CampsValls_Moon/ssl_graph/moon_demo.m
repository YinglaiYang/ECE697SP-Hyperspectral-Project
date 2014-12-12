
%=====================================================================
% Generate the toy data 
%=====================================================================
clear all; 
close all; 

[X, Y] = moon;

f = figure(1);
set(f, 'Position',  [0 114 1200 420]); 
subplot(121);
Z = Y(:, 1) - Y(:, 2); 
label_1 = find(Z > 0);
label_2 = find(Z < 0);

b = plot(X(label_1,1), X(label_1,2),'rd', 'MarkerSize', 25); hold on;
set(b, 'MarkerFace', 'r','LineWidth',1.5);
c = plot(X(label_2,1), X(label_2,2),'g<', 'MarkerSize', 25); 5
set(c, 'MarkerFace', 'g','LineWidth',1.5);

a = plot(X(:, 1), X(:, 2), 'b.' , 'MarkerSize', 15); 
p=legend([b, c], '+1 class',  '-1 class'); set(p,'Fontsize',20);
hold off;
title('Toy Data (Two Moons)');

%=====================================================================
% Classify the toy data
%=====================================================================
sigma = 0.1; alpha = 0.99;
C = semi(X,Y, sigma,alpha); 


subplot(122);
pc = find(C == 1);
nc = find(C == 2);

b = plot(X(label_1,1), X(label_1,2),'rd', 'MarkerSize', 25); hold on;
set(b, 'MarkerFace', 'r','LineWidth',1.5);
c = plot(X(label_2,1), X(label_2,2),'g<', 'MarkerSize', 25); 
set(c, 'MarkerFace', 'g','LineWidth',1.5);

d = plot(X(pc,1), X(pc,2),'rd', 'MarkerSize', 10); hold on;
set(d, 'LineWidth',1.5);
e = plot(X(nc,1), X(nc,2),'g<', 'MarkerSize', 10); 
set(e, 'LineWidth',1.5);

hold off;
title('classification result');
