function [window] = getWindow(img, windowSize)
%according to the windowsize and original matrix to get different window
%get the size of the whole image
[a, b, d] = size(img);
%get the size of the window
width = windowSize(1);
height = windowSize(2);
window = zeros(1, 4);
flag = 0;

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
        if(flag == 1 || (i < ceil(a/width) &&  j < ceil(b/height)))
            window = [(i-1)*width+1, (j-1)*height+1, i*width, j*height];
        end     
        if(flag>1 && j == ceil(b/height))
            window = [(i-1)*width+1, b-height, i*width, b];
        end
        if(flag>1 && i == ceil(a/width))
            window = [a-width, (j-1)*height+1, a, j*height];
        end
        if(flag == 4 && i == ceil(a/width) &&  j == ceil(b/height))
            window = [a-width, b-height, a, b];
        end
        disp(['flag: ' num2str(flag)]);
        disp(['window: ' num2str(window)]);
      
    end
end
end

