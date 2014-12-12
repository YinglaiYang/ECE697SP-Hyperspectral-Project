clear, clc, close all;
runtime = tic;
load hsi;
load mapTrain;
load mapTest;

%% Parameters
[a, b, d] = size(hsi);
c = unique(mapTrain);
c = c(length(c));
%define the size of the window
height = 10;
width = 10;
window = zeros(1, 4);
flag = 0;
F_star = zeros(a, b, c);

alpha = 0.1;
sigma = 0.6;
m_fraction = 1e-2;

get_f_star = @(Y, X) sesuGraph(Y, X, alpha, sigma, m_fraction);

switch num2str(mod(a, height)~=0)+num2str(mod(a, width) ~= 0)
    case '00'
        flag = 1; 
    case '01'
        flag = 2;
    case '10'
        flag = 3;
    otherwise
        flag = 4;
end

for i = 1:ceil(a/height)
    for j = 1:ceil(b/width)
        if(i < ceil(a/height) &&  j < ceil(b/width))
            window = [(i-1)*height+1, (j-1)*width+1, i*height, j*width];
            [~, newX, newY] = getSubwindowData(mapTrain, hsi, window);
            f_star = get_f_star(newY, newX);
            F_star((i-1)*height+1:i*height, (j-1)*width+1:j*width,:) = ...
                       predictionMapFromFstar(f_star(1:width*height,:), width, height);
            disp(['window: ' num2str(window)]);
        end     
        if( i < ceil(a/height) && j == ceil(b/width))
            window = [(i-1)*height+1, b-width+1, i*height, b];
            [~, newX, newY] = getSubwindowData(mapTrain, hsi, window);
            f_star = get_f_star(newY, newX);
            f_star = reshape(f_star(1:height*width,:), height, width, c);
            F_star((i-1)*height+1:i*height, (j-1)*width+1:b,:) = f_star(:, width-b+(j-1)*width+1:width, :);
            disp(['window: ' num2str(window)]);
        end
        if(i == ceil(a/height) && j < ceil(b/width))
            window = [a-height+1, (j-1)*width+1, a, j*width];
            [~, newX, newY] = getSubwindowData(mapTrain, hsi, window);
            f_star = get_f_star(newY, newX);
            f_star = reshape(f_star(1:height*width,:), height, width, c);
            F_star((i-1)*height+1:a, (j-1)*width+1:j*width, :) = f_star(height-a+(i-1)*height+1:height,:,:);
            disp(['window: ' num2str(window)]);
        end
        if(i == ceil(a/height) &&  j == ceil(b/width))
            window = [a-height+1, b-width+1, a, b];          
            [~, newX, newY] = getSubwindowData(mapTrain, hsi, window);
            f_star = get_f_star(newY, newX);
            f_star = reshape(f_star(1:height*width,:), height, width, c);
            F_star((i-1)*height+1:a, (j-1)*width+1:b, :) = f_star(height-a+(i-1)*height+1:height, width-b+(j-1)*width+1:width,:);
            disp(['window: ' num2str(window)]);
        end
        
    end
end
toc(runtime)
