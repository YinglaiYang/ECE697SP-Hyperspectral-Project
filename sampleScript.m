% sampleScript.m
% ECE 697SP Fall 2014 Final Project
% Classification of the mineral map in cuprite, Nevada
% a sample code
% In this project, you will work on a multi-class classification problem of
% real hyperspectral data.
% Have fun!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load hsi
[line,sample,band] = size(hsi);
% hsi.mat contains two variables: hsi and wavelength
% hsi: hyperspectral data [line x sample x band]
%      hsi(:,:,i) shows the image of the i th band
%      hsi(i,j,:) shows the observed spectrum at location (i,j). In this
%      image, spectra are already converted into reflectance
% wavelength : wavelength corresponding to hsi. the length is the same as
% that of the third dimension of hsi.
load mapTrain
% mapTrain.mat contains one variable: mapTrain
% mapTrain is a classification map you use for training classifiers. The
% size is the same as the first and second dimensions of the hsi, namely
% [sample x line]. Each pixel value means the class label at that location.
% You have 19 classes total, and the if the pixel value is i, which means
% that point belongs to the i th class. Zeros mean the unknown point. You
% can also use the unknown points for classification. Some of the unknown 
% points are used for testing, but you do not know their labels.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% show images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if you want to show image at one band, you can use the code below.
im1 = squeeze(hsi(:,:,1));
imagesc(im1);
colormap(gray);
axis equal;
axis tight;
% you can also use commands, imshow (or image).
imshow(im1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% show spectra
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if you wanna show spectra, use commands below
% you can see the shapes of spectra. The absorption features are usually
% used for detecting minerals
spc = squeeze(hsi(1,1,:));
figure;
plot(wavelength,spc,'r-');
axis tight;
ylim([0,1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% show classification map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% you can also show classification map with imagesc
imagesc(mapTrain);
colormap(jet);
axis equal;
axis tight;
% different colors mean different classes.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sample classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% As an example, we perform nearest neighbor classification
% put cvKnn.m and cvEucdist.m in the same directory.

% before doing classification, we recommend you to ignore the unknown
% points. Or you can use the unknown points in a semi-supervised way if you
% want.
nonZeros = find(mapTrain>0);
X = reshape(hsi,sample*line,band);
t = mapTrain(nonZeros);
X = X(nonZeros,:);

% nearest neighbor classifier. Showing training errors
k=10;
pred = knnclassify(X,X,t,k,'euclidean');
accu = mean(pred(:)==t(:))*100;
fprintf('Accuracy of the kNN with k=%d is %f [%%].\n',k,accu);
% You will get around 74 [%] accuracy.

% You can also do multi-class SVM using libsvm 
% at http://www.csie.ntu.edu.tw/~cjlin/libsvm/#download
% if you do not know how to install, please aske me.
% you can perform linear or kernel SVMs.
model = svmtrain(t,X,'-s 0 -t 0 -c 10');
pred = svmpredict(t,X,model);
accu = mean(pred(:)==t(:))*100;
fprintf('Accuracy of the linear CSVM is %f [%%].\n',accu);
% You will get around 75.0 [%] accuracy.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cross validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can perform cross validation to select best features
% five fold cross validation

% first separate training data into 5 different sets
kfolds = crossvalind('Kfold',t,5);

% perform cross validation of kNN for different parameters k 
accuCV = zeros(5,10);
for k=1:10
    for i = 1:5
        idxTest = kfolds==i;
        idxTrain = kfolds~=i;
        Xtrain = X(idxTrain,:); tTrain = t(idxTrain);
        Xtest = X(idxTest,:); tTest = t(idxTest);

        pred = knnclassify(Xtest,Xtrain,tTrain,k,'euclidean');
        accu = sum(pred(:)==tTest(:));
        % because the number of each folds is slightly diffrent, we count
        % how many points are misclassified and divide the misclassified
        % number by the total number in the end.
        accuCV(i,k) = accu;
    end
end
accuCV = sum(accuCV,1)/length(t);
figure;
plot(1:10,accuCV);
title('Cross validation accuracy');
ylabel('Accuracy'); xlabel('k');

% now you can perform and test your novel algorithm on this data!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% submission of the classification results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Once you apply your novel method and produce the predicted classification
% map, submit a classified image. You can use the same format as mapTrain.
% we show sample commands to produce the image. 

% extract test training and test points
X = reshape(hsi,sample*line,band);
Zeros = find(mapTrain==0);
Xtest = X(Zeros,:);
nonZeros = find(mapTrain>0);
tTrain = mapTrain(nonZeros);
Xtrain = X(nonZeros,:);

% prediction by kNN
k=8;
pred = knnclassify(Xtest,Xtrain,tTrain,k,'euclidean');
mapPredict = mapTrain;
% assign predicted classes for a whole image other than training points.
mapPredict(Zeros) = pred;

save mapPredict.mat mapPredict;

% submit mapPredict.mat
% Now you have done!



