% Name: Connor Thompson
% SID: 107554044
% Date: 4/19/2020
% Assignment Number: Final Project
% Course Number: CSCI 4830 - Computer Vision
% Instructor: Dr. Fleming

function [] = displayResults()
    % Results from running main.m on my 9 selected images from BSDS
    filenames = {'girlRowingResults.mat.png', '216053.mat.png', '216041.mat.png', '187071.mat.png', '126039.mat.png', '76002.mat.png', '65019.mat.png', '65010Results.mat.png', '42044.mat.png'};
    for i = 1:size(filenames, 2)
        figure()
        name = "./results/mPb-border-Results/" + filenames{1, i};
        temp = imread(name);
        factor = 255 / max(temp(:));
        temp = uint8(temp .* factor);
        imshow(temp);
        colormap jet
        colorbar
    end
end