%% Recreates Indian Pines Map Used in "Semi-Supervised Graph-Based Hyperspectral Image Classification"
% Not a 1:1 reproduction, since it could not be ensured that the same bands
% are used. This dataset was found on http://www.ehu.es/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines
% Date of access: 11th Dec 2014

%% References
% - [1]: Semi-Supervised Graph-Based Hyperspectral Image Classification;
% Camps-Valls et al.
% - [2]: Indian Pines dataset; http://www.ehu.es/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines

%% Load dataset
load('Indian_pines_corrected.mat')
load('Indian_pines_gt.mat') % Groundtruth

%% Get the 68x86 subset that was used in [1]
cols = 27:94;
rows = 31:116;

subip_hsi = indian_pines_corrected(rows,cols,:);
subip_gt = indian_pines_gt(rows,cols);

%% Create training and test map
nonzeros = subip_gt > 0;

t = subip_gt(nonzeros);

cv_ind = crossvalind('Kfold', t, 5); % use 20% as training set and other 80% as test set

trainIDX = cv_ind == 1;
testIDX  = cv_ind ~= 1;

nonzeros_pos = find(nonzeros);

ipMapTrain = zeros(numel(subip_gt), 1);
ipMapTest  = zeros(numel(subip_gt), 1);

train_pos = nonzeros_pos(trainIDX);
test_pos  = nonzeros_pos(testIDX);

ipMapTrain(train_pos) = subip_gt(train_pos);
ipMapTest(test_pos)   = subip_gt(test_pos);

ipMapTrain = reshape(ipMapTrain, size(subip_gt));
ipMapTest  = reshape(ipMapTest, size(subip_gt));

labels = unique(ipMapTrain(ipMapTrain~=0));

for lbel = 1:length(labels)
    label = labels(lbel);
    
    ipMapTrain(ipMapTrain == label) = lbel;
    
    ipMapTest(ipMapTest == label) = lbel;
end

save('ipMapTrain.mat', 'ipMapTrain');
save('ipMapTest.mat',  'ipMapTest');
save('subip_hsi.mat', 'subip_hsi');