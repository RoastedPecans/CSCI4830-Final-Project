% Name: Connor Thompson
% SID: 107554044
% Date: 4/19/2020
% Assignment Number: Final Project
% Course Number: CSCI 4830 - Computer Vision
% Instructor: Dr. Fleming

% This script is used to train our mPb weights based on ground-truth
% segmentations from Berkeley
% https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500.
%
% It uses a 50/50 chance to either try random new weights or try a slight variation
% of the current best global weights

clc;
clear all;

% Result files from main.m
filenames = {'girlRowingResults.mat', '42044.mat', '65010Results.mat', '65019.mat', '76002.mat', ...
    '126039.mat', '187071.mat', '216041.mat', '216053.mat'};

% Ground Truth segmentation files from Berkeley
truthFiles = {'girlRowingGroundTruth.mat', '42044GroundTruth.mat', '65010GroundTruth.mat', '65019GroundTruth.mat', '76002GroundTruth.mat', ...
    '126039GroundTruth.mat', '187071GroundTruth.mat', '216041GroundTruth.mat', '216053GroundTruth.mat'};

% Initialize some variables
truths = cell(size(truthFiles, 2));  % Holds averaged ground-truths
bestImageScores = cell(size(truthFiles, 2));  % Holds min SSD per image
bestImages = cell(size(truthFiles, 2));  % Holds copy of best image 
bestImageWeights = cell(size(truthFiles, 2));  % Each Image's best weights

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
    
    % Normalize to 0-1
    truths{i} = mat2gray(tempImg);
    bestImageScores{i} = 1000000000;
end

% Initalize some variables
bestGlobalWeights = zeros([1, (4 * 8)]);
iterations = 5000;
globalMin = 10000000000;
rate = 0.0005;
randomFlag = false;

% Load files if applicable so we can leave off training where we were
load('./results/matlab-files/bestImageCellArrays.mat');
load('./results/matlab-files/bestGlobalWeights.mat');

% How many iterations we should try to generate random weights for
for i = 1:iterations
    
    % 50/50 Chance to randomly initalize or try modified best
    if rand() > 0.5
        disp("Iteration: " + i + "    Initialize New Random Weights");
        % We have 8 orientation at 4 scales/features
        weights = rand([1, (4 * 8)]);
        randomFlag = true;
    else
        disp("Iteration: " + i + "    Try Close Weights");
        % rand() - 0.5 gives us random range from [-0.5, 0.5]
        % then multiply by rate * rand() because we want a small, random
        % number because why not!
        weights = bestGlobalWeights + ((rand([1, 32]) - 0.5) .* (rate * rand()));
    end
    roundSSD = 0;
    
    % For each file...
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
               newResults(:, :, o) = newResults(:, :, o) + weights(1, counter) * results(:, :, s, o);
               counter = counter + 1;
           end
       end
       
       % Transform mPb(x, y, theta) to mPb(x, y)
       pb = mPb(newResults);

       % Normalize + Invert to match our ground truths
       newImageNorm = mat2gray(pb);
       bounds = find(newImageNorm < 0.25);
       newImageNorm(bounds) = 1;
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
           bestImageWeights{j} = weights;
           save('./results/matlab-files/bestImageCellArrays.mat', 'bestImageScores', 'bestImages', 'bestImageWeights');
       end
       
       roundSSD = roundSSD + ssd;
       
    end
    
    if rate > 0.0000001
        rate = rate - 0.0000001;
    end
    
    % Check if we have new global best
    if roundSSD < globalMin
        globalMin = roundSSD;
        disp("New Global Best: " + globalMin);
        bestGlobalWeights = weights;
        save('./results/matlab-files/bestGlobalWeights.mat', 'bestGlobalWeights', 'globalMin');
        imshow([bestImages{2}, bestImages{3}, bestImages{8}, bestImages{9}]);
%         for x = 1:9
%             filename = filenames{x};
%             name = "./results/mPb-Results/" + filename + ".png";
%             imwrite(bestImages{x}, name); 
%         end
        
        % If a random update gave us our new best, reset rate
        if randomFlag
            rate = 0.0005;
        end
    else
    end
    randomFlag = false;
end