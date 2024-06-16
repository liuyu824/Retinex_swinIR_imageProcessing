clc;clear;close all;

resultPath = './dataset/resultTest/';  % ���
targetPath = './dataset/targetTest/';  % Ŀ��

resultImagesPath = dir(strcat(resultPath, '*.png'));
targetImagesPath = dir(strcat(targetPath, '*.png'));

n = length(targetImagesPath);
psnr_total = 0;

for i=1:n
   
    resultImagePath = fullfile(resultPath, resultImagesPath(i).name);
    targetImagePath = fullfile(targetPath, targetImagesPath(i).name);
    
    resultImage = imread(resultImagePath);
    targetImage = imread(targetImagePath);
    
    psnr_total = psnr_total + compute_psnr(resultImage, targetImage);
    
end

fprintf('Average PSNR: %f\n', psnr_total/n);

