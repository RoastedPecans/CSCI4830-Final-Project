% Name: Connor Thompson
% SID: 107554044
% Date: 4/19/2020
% Assignment Number: Final Project
% Course Number: CSCI 4830 - Computer Vision
% Instructor: Dr. Fleming

function [smallFeatures, mediumFeatures, largeFeatures] = getFeatures(img)
    [height, width] = size(img);
    
    % Each will have 8 oriented even and odd symmetric Gaussian Derivative
    % Filters and one DOG Filter (See Figure 5 in Paper)
    smallFilters = zeros(19, 19, 16);
    mediumFilters = zeros(27, 27, 16);
    largeFilters = zeros(39, 39, 16);
    
    % Create arrays to hold results from convolving filters
    smallFeatures = zeros(height, width, 16);
    mediumFeatures = zeros(height, width, 16);
    largeFeatures = zeros(height, width, 16);
    
    % Create actual filter bank -- DO NOT CHANGE THESE PARAMS UNLESS YOU
    % KNOW HOW TO ADJUST EVERYTHING ELSE
    % From https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
    % Specifically, the "Filters" Folder
    filters = fbCreate(8, 1, 3, sqrt(2), 3);
    
    % Move filters from generated cell array to my arrays
    for i = 1:16
        smallFilters(:, :, i) = filters{i, 1};
        mediumFilters(:, :, i) = filters{i, 2};
        largeFilters(:, :, i) = filters{i, 3};
    end
    
    % Convolve each filter with our image
    for i = 1:16
        smallFeatures(:, :, i) = conv2(img, smallFilters(:, :, i), 'same');
        mediumFeatures(:, :, i) = conv2(img, mediumFilters(:, :, i), 'same');
        largeFeatures(:, :, i) = conv2(img, largeFilters(:, :, i), 'same');
    end
    
    return
end