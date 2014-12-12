%%Function to get the Y matrix
function [Y] = getLabelMatrixY(mapTraining, c)
%get all the classes for the training set

%get the size of the training set
N_samples = numel(mapTraining);

%build Y matrix with the coressponding size
Y = zeros(N_samples, c);

for label=1:c
    labelLocation = (mapTraining == label);
    
    Y(labelLocation,label) = 1;
end
            
end