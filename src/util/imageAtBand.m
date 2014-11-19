function imageAtBand( spectralCube, band )
%SHOWIMAGEATBAND Summary of this function goes here
%   Detailed explanation goes here
imageMatrix = squeeze(spectralCube(:,:,band));
% imagesc(imageMatrix);
colormap(gray);
axis equal;
axis tight;
imshow(imageMatrix);
end

