% Name: Connor Thompson
% SID: 107554044
% Date: 4/19/2020
% Assignment Number: Final Project
% Course Number: CSCI 4830 - Computer Vision
% Instructor: Dr. Fleming

% This script is essentially the same as train.m, but I was going to modify
% it to train my gPb weights instead of my mPb weights. However, I ran into
% issues when trying to use sPb (part of gPb) so I ended up just using this
% script to train a new set of weights where there is a 15 pixel border of
% black around both the truth and test images to eliminate some of the
% "funkiness" that was occurring in the original training. My hope is that
% by eliminating the weird noise on the edge of my mPb images + the truth
% images, I can get my weights to better converge directly on the ground
% truth contours rather than just slightly adjusting the noise on the
% outside of the image.

% Uses 50/50 chance to either try random new weights or slightly modify
% previous best weights +- some small num.

clc;
clear all;

% Results from running main.m on my 9 selected images from BSDS
filenames = {'girlRowingResults.mat', '42044.mat', '65010Results.mat', '65019.mat', '76002.mat', ...
    '126039.mat', '187071.mat', '216041.mat', '216053.mat'};

% Ground-Truth Segmentations from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
truthFiles = {'girlRowingGroundTruth.mat', '42044GroundTruth.mat', '65010GroundTruth.mat', '65019GroundTruth.mat', '76002GroundTruth.mat', ...
    '126039GroundTruth.mat', '187071GroundTruth.mat', '216041GroundTruth.mat', '216053GroundTruth.mat'};

% Create some arrays to hold data
truths = cell(size(truthFiles, 2));  % Holds averaged ground-truths
bestImageScores = cell(size(truthFiles, 2));  % Min SSD per image
bestImages = cell(size(truthFiles, 2));  % Copy of best image
bestImageWeights = cell(size(truthFiles, 2));  % Weights used per image

% Generate Average Truth Images from BSDS data
for i = 1:size(truthFiles, 2)
    % Load in ground truth file
    fName = "./images/groundTruth/" + truthFiles{1, i};
    file = load(fName);
    tempImg = zeros(size(file.groundTruth{1, 1}.Boundaries));
    numTruths = size(file.groundTruth, 2);
    
    % Sum each human segmentation
    for j = 1:numTruths
        tempImg = tempImg + file.groundTruth{1, j}.Boundaries;
    end
    
    % Add 15 pixel border of black (see line 1 comment)
    size1 = size(tempImg, 1);
    size2 = size(tempImg, 2);
    tempImg(1:15, :) = 0;
    tempImg(:, 1:15) = 0;
    tempImg(size1-15:size1, :) = 0;
    tempImg(:, size2-15:size2) = 0;
    
    % Normalize Image to 0-1
    truths{i} = mat2gray(tempImg);
    
    bestImageScores{i} = 1000000000;
end

% Initalize Some Stuff
bestGlobalWeights = zeros([1, (4 * 8)]);
iterations = 5000;
globalMin = 10000000000;
rate = 0.0005;
randomFlag = false;

% Load previous data if we have it so we have our previous best scores
load('./results/matlab-files/bestImageCellArraysgPb.mat');
load('./results/matlab-files/bestGlobalWeightsgPb.mat');

% How many iterations we should try to generate random weights for
for i = 1:iterations
    
    % 50/50 Chance to randomly initalize or try modified best
    if rand() > 0.5
        disp("Iteration: " + i + "    Initialize New Random Weights");
        % We have 8 orientation at 4 scales/features
        mPbweights = rand([1, (4 * 8)]);
        randomFlag = true;
    else
        disp("Iteration: " + i + "    Try Close Weights");
        % rand() - 0.5 gives us random range from [-0.5, 0.5]
        % then multiply by rate * rand() because we want a small, random
        % number because why not!
        mPbweights = bestGlobalWeights + ((rand([1, 32]) - 0.5) .* (rate * rand()));
    end
    roundSSD = 0;
    
    % For each results file (from main.m)
    for j = 1:size(filenames, 2) 
       filename = "./results/matlab-files/" + filenames{j};
       
       % Load results into "results" variable
       load(filename);
       
       newResults = zeros([size(results, 1), size(results, 2), 8]);

       % Calculate weighted gradients (mPb(x, y, theta) from paper)
       %
       % Sums each orientation we have at each scale so we're left
       % with a 3D matrix where the first two are the image, and the
       % third the summed orientation across scalesw
       counter = 1;
       for s = 1:4
           for o = 1:8
               newResults(:, :, o) = newResults(:, :, o) + mPbweights(1, counter) * results(:, :, s, o);
               counter = counter + 1;
           end
       end
       
       % Transform mPb(x, y, theta) to mPb(x, y)
       pb = mPb(newResults);
       size1 = size(pb, 1);
       size2 = size(pb, 2);
       
       % This is where I ran into issues with sPb
       % Tried to use sPb code from
       % https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500
       % to simplify things, but with no luck
       %sb = spectralPb(pb, size(pb), '', 16);

       % Normalize + Add 15px Borders + Invert to match our ground truths
       newImageNorm = mat2gray(pb);
       bounds = find(newImageNorm < 0.25);
       newImageNorm(bounds) = 1;
       newImageNorm(1:15, :) = 1;
       newImageNorm(:, 1:15) = 1;
       newImageNorm(size1-15:size1, :) = 1;
       newImageNorm(:, size2-15:size2) = 1;
       newImageNorm = imcomplement(newImageNorm);
       
       %imshow([truths{j}, newImageNorm]);
       %x = input('x', 's');
       
       % Calculate SSD to minimize
       ssd = immse(newImageNorm, truths{j});
       
       % Check + Update each individual image so we have the best of each
       if ssd < bestImageScores{j}
           disp("Image " + j + " has new best score of: " + ssd);
           bestImageScores{j} = ssd;
           bestImages{j} = newImageNorm;
           bestImageWeights{j} = mPbweights;
       end
       
       roundSSD = roundSSD + ssd;
       
    end
    
    if rate > 0.0000001
        rate = rate - 0.0000001;
    end
    
    % Check if we have new global best
    if roundSSD < globalMin
        iterationsWithoutNewBest = 0;
        globalMin = roundSSD;
        disp("New Global Best: " + globalMin);
        bestGlobalWeights = mPbweights;
        save('./results/matlab-files/bestImageCellArraysgPb.mat', 'bestImageScores', 'bestImages', 'bestImageWeights');
        save('./results/matlab-files/bestGlobalWeightsgPb.mat', 'bestGlobalWeights', 'globalMin');
        %imshow([bestImages{2}, bestImages{3}, bestImages{8}, bestImages{9}]);
        %x = input('x', 's');
        
        % Uncomment to save best images to directory
        %for x = 1:9
        %    filename = filenames{x};
        %    name = "./results/mPb-border-Results/" + filename + ".png";
        %    imwrite(bestImages{x}, name); 
        %end
        
        % If random initalization gave us new best, reset learning rate
        if randomFlag
            rate = 0.0005;
        end
    else
    end
    randomFlag = false;
end