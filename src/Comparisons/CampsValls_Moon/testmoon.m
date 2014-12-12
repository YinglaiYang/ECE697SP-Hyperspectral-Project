clc, close all, clear;

moon_demo;

z2 = Z;
z2(find(z2==-1)) =2;
%% Parameters
alpha = 0.99;
sigma = 0.1;

nystroemFractions = [0.1 0.3 0.5 0.7 1.0];

for nf=1:length(nystroemFractions)
    nystroemFraction = nystroemFractions(nf); % 1percent

    RSVD.k_fraction = 1.0;
    RSVD.p = 10;
    RSVD.q = 3;

    tic;
    %% Perform classification
    get_f_star = @(Y, X) sesuGraph_preciseD(Y, X, alpha, sigma, nystroemFraction, RSVD);
    Y = getLabelMatrixY(z2, 2);
    f_star = get_f_star(Y, X);
    predict = predictionMapFromFstar(f_star,1,200);
    %mapPredict = windowedClassifierRSVD(X, z2, alpha, sigma, nystroemFraction, windowHeight, windowWidth);

    %% Given data
    f = figure;
    %% Classification plot
    pc = find(predict == 1);
    nc = find(predict == 2);

    b = plot(X(label_1,1), X(label_1,2),'rd', 'MarkerSize', 25); hold on;
    set(b, 'MarkerFace', 'r','LineWidth',1.5);
    c = plot(X(label_2,1), X(label_2,2),'g<', 'MarkerSize', 25); 
    set(c, 'MarkerFace', 'g','LineWidth',1.5);

    d = plot(X(pc,1), X(pc,2),'rd', 'MarkerSize', 10); hold on;
    set(d, 'LineWidth',1.5);
    e = plot(X(nc,1), X(nc,2),'g<', 'MarkerSize', 10); 
    set(e, 'LineWidth',1.5);

    hold off;
    title(sprintf('Classification result with our code; m=%.0d \\times n', nystroemFraction));
    figureName = sprintf('toymoon_nystFactor_%d_percent', nystroemFraction*100);
    savefig(figureName);
    saveas(gcf, figureName, 'png');
end