%%Function to get the Y matrix
function [Y] = getLabelMatrixY(mapTraining)
%get all the classes for the training set
c = unique(mapTraining);
%get the size of the training set
[a, b] = size(mapTraining);
%build Y matrix with the coressponding size
Y = zeros(a*b,c(length(c)));
for j = 1:b
    for i = 1:a
        if(mapTraining(i,j) ~= 0)
            Y(sub2ind([a,b], i, j), mapTraining(i,j)) = 1;
        end
    end
end
            
end