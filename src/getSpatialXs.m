%% function to get the spatial xs
function [Xs] = getSpatialXs(img)  %use the original image hsi
[a,b,c]= size(img);
Xs = zeros(a,b,c);
A = zeros(a,b);
B = ones(3,3)/9;
for i = 1:c
   A = img(:,:,i);
   C = conv2(A,B,'same');
   Xs(:,:,i) = C(:,:);
end
end