% Name: Connor Thompson
% SID: 107554044
% Date: 4/19/2020
% Assignment Number: Final Project
% Course Number: CSCI 4830 - Computer Vision
% Instructor: Dr. Fleming

% Runs the Multiscale-Pb Algorithm on the inputted data
% Take the max along orientation index (y, x, o)
% Takes in mPb(x, y, theta)
% Returns mPb(x, y)
function [image] = mPb(dataMatrix)
    [height, width, orientations] = size(dataMatrix);
    image = zeros([height, width]);
    
    % Take max value of each orientation for each pixel
    for i = 1:height
        for j = 1:width
            image(i, j) = max(dataMatrix(i, j, :));
        end
    end
    
    return
end
