clear, clc, close all;
runtime = tic;
load hsi;
load mapTrain;
load mapTest;

[a, b, d] = size(hsi);
c = unique(mapTrain);
c = c(length(c));
%define the size of the window
width = 165;
height = 125;
window = zeros(1, 4);
flag = 0;
F_star = zeros(a, b, c);
switch num2str(mod(a, width)~=0)+num2str(mod(a, height) ~= 0)
    case '00'
        flag = 1; 
    case '01'
        flag = 2;
    case '10'
        flag = 3;
    otherwise
        flag = 4;
end

for i = 1:ceil(a/width)
    for j = 1:ceil(b/height)
        if(i < ceil(a/width) &&  j < ceil(b/height))
            window = [(i-1)*width+1, (j-1)*height+1, i*width, j*height];
            [newX, newY] = matrix(mapTrain, hsi, window);
            f_star = sesuGraph(newY, newX, 0.01, @(x1,x2) RBF_kernel(x1, x2, 0.7), 1e-2, 0.2);
            f_star = reshape(f_star(1:width*height,:), width, height, c);
            F_star((i-1)*width+1:i*width, (j-1)*height+1:j*height,:) = f_star;
            disp(['window: ' num2str(window)]);
        end     
        if( i < ceil(a/width) && j == ceil(b/height))
            window = [(i-1)*width+1, b-height+1, i*width, b];
            [newX, newY] = matrix(mapTrain, hsi, window);
            f_star = sesuGraph(newY, newX, 0.01, @(x1,x2) RBF_kernel(x1, x2, 0.7), 1e-2, 0.2);
            f_star = reshape(f_star(1:width*height,:), width, height, c);
            F_star((i-1)*width+1:i*width, (j-1)*height+1:b,:) = f_star(:, height-b+(j-1)*height+1:height, :);
            disp(['window: ' num2str(window)]);
        end
        if(i == ceil(a/width) && j < ceil(b/height))
            window = [a-width+1, (j-1)*height+1, a, j*height];
            [newX, newY] = matrix(mapTrain, hsi, window);
            f_star = sesuGraph(newY, newX, 0.01, @(x1,x2) RBF_kernel(x1, x2, 0.7), 1e-2, 0.2);
            f_star = reshape(f_star(1:width*height,:), width, height, c);
            F_star((i-1)*width+1:a, (j-1)*height+1:j*height, :) = f_star(width-a+(i-1)*width+1:width,:,:);
            disp(['window: ' num2str(window)]);
        end
        if(i == ceil(a/width) &&  j == ceil(b/height))
            window = [a-width+1, b-height+1, a, b];          
            [newX, newY] = matrix(mapTrain, hsi, window);
            f_star = sesuGraph(newY, newX, 0.01, @(x1,x2) RBF_kernel(x1, x2, 0.7), 1e-2, 0.2);
            f_star = reshape(f_star(1:width*height,:), width, height, c);
            F_star((i-1)*width+1:a, (j-1)*height+1:b, :) = f_star(width-a+(i-1)*width+1:width, height-b+(j-1)*height+1:height,:);
            disp(['window: ' num2str(window)]);
        end
        
    end
end
toc(runtime)
