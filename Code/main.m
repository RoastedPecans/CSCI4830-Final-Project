clc;
clear all;

% Name: Connor Thompson
% SID: 107554044
% Date: 4/19/2020
% Assignment Number: Final Project
% Course Number: CSCI 4830 - Computer Vision
% Instructor: Dr. Fleming

%% Before running this software please note the pre-requisites:
% Requires Signal Processing Toolbox
% Must add utilities folder to Matlab Path (MatLab should prompt you to do
% this!)

%% This Script runs the final project -- we will only support black and white images

% Initialize some variables we will need
matlabFilename = "./results/matlab-files/65019.mat";  % Location of matlab file we want to save/load from
imgFilename = "./images/65019.jpg";  % Location of image file to use
img = uint8(rgb2gray(imread(imgFilename)));
[height, width] = size(img);
%angles = [0, 0.5236, 1.0472, 1.5708, 2.0944, 2.6180];
angles = [0, 0.3927, 0.7854, 1.1781, 1.5708, 1.9635, 2.3562, 2.7489];
sigma = 10;
featureScales = [20, 28, 40];
numScales = size(featureScales, 2);
numAngles = size(angles, 2);

% If you just want to display results
%displayResults();

% Save/Load functionality
saveRun = false;  % Set to true if you wish to save the results from this run
loadRun = true;  % Set to true if you wish to load previous processed data

if ~loadRun

    %% Get Texture Information From Image
    % Generates 16 filters + convolves them with our image at 3 different
    % scales. Returns results of convolutions.
    % We will collapse these 48 filters into a single texton id
    % with clustering.
    % 16 oriented bars
    disp("Creating Filter Bank + Convolving with Image");
    [smallFeatures, mediumFeatures, largeFeatures] = getFeatures(img);

    
    %% Compute Texton ID's Using K-Means
    % Input Matrix: Rows of X = pixels, columns = features
    % Permute to make reshape into row-major order, reshape to [height *
    % width, 16] before running K=32 Means clusters up to 250 iterations.
    %
    % In addition, run K-Means with 5 different initial starting points,
    % parallelize these 5 different runs
    %
    % This assigns each pixel in our image a cluster based on the textures
    % associated with that pixel
    disp("Creating Textons based on Filter Convolutions");
    textonsScale1 = kmeans(reshape(permute(smallFeatures(:, :, :), [2, 1, 3]), [height * width, 16]), 32, 'MaxIter', 250, 'Replicates', 5, 'Options', statset('UseParallel', 1));
    textonsScale2 = kmeans(reshape(permute(mediumFeatures(:, :, :), [2, 1, 3]), [height * width, 16]), 32, 'MaxIter', 250, 'Replicates', 5, 'Options', statset('UseParallel', 1));
    textonsScale3 = kmeans(reshape(permute(largeFeatures(:, :, :), [2, 1, 3]), [height * width, 16]), 32, 'MaxIter', 250, 'Replicates', 5, 'Options', statset('UseParallel', 1));
    
    %% Reshape K-Means Cluster Results from 1xN back to our original Image
    % Had to do this manually since MatLab reshape doesn't work the right
    % way...
    %
    % We also add our raw intensity into our features, giving us 3 texture maps
    % and 1 intensity image.
    
    scales = zeros(height, width, numScales + 1);
    
    % Go through each of the resulting clusters from K-Means and manually
    % "tape" the rows of the image back together one by one
    for s = 1:numScales + 1
        counter = 1;
        for i = 1:height
            if s == 1
                scales(i, :, s) = textonsScale1(counter:counter+width-1)';
            elseif s == 2
                scales(i, :, s) = textonsScale2(counter:counter+width-1)';
            elseif s == 3
                scales(i, :, s) = textonsScale3(counter:counter+width-1)';
            end
            counter = counter + width;
        end
        % Add raw intensity information to our data matrix
        if s == 4
            scales(:, :, s) = img;
        end
    end

    % Display our original image along with our k-means texture clusters
    imshow([img, scales(:, :, 1), scales(:, :, 2), scales(:, :, 3)], [1, 32]);
    title("Results of K-Means Clustering on Our Textures");
    x = input('x', 's');

    %% Pre-rotate each Texture Map + Intensity feature
    % This saves us a lot of computation time + simplifies things a lot
    % later on when we start comparing differences of areas at various
    % angles.
    %
    % See Appendix of Research Paper for more details covered by the authors
    disp("Pre-Rotating our Texton Mappings + Intensity Image");
    rotatedImages = zeros([height, width, numScales + 1, numAngles]);
    for i = 1:numScales + 1
        for j = 1:numAngles
            rotatedImages(:, :, i, j) = imrotate(scales(:, :, i), -rad2deg(angles(j)), 'crop');
            %imshow(rotatedImages(:, :, i, j), [0, 32]);
            %x = input('x', 's');
        end
    end

    
    %% Go through each rotated feature and Find Histogram Differences with different size windows
    % This is the largest portion of code
    
    % Will hold our results from the histogram differences
    rotatedResults = zeros([height, width, numScales + 1, numAngles]);
    
    disp("Calculating Gradients for Each Feature...");
    % For each scale we want to get information from...
    for i = 1:numScales + 1
        disp("Feature " + i + " of " + (numScales + 1));

        radius = 3;

        % If on the intensity feature, set radius to 3
        if i ~= numScales + 1
            radius = (featureScales(i) - 2) / 2;  % How far we need to go in each direction around center pixel
        end
        
        % For each angle we want to get information from...
        for j = 1:numAngles
            % Get the pre-rotated texture map we will be working with
            tempImg = rotatedImages(:, :, i, j);
            
            % Go through each image in our pre-rotated texutre map
            for y = radius + 1:height - radius
                for x = radius + 1:width - radius

                    % Get Bounds
                    % -radius thru current column for left half
                    % current column + 1 thru radius + 1 for right half
                    leftBound = uint16(x - radius);
                    rightBound = uint16(x + radius + 1);
                    topBound = uint16(y - radius);
                    bottomBound = uint16(y + radius + 1);

                    % Make ints so matlab stops yelling at me :(
                    yInt = uint16(y);
                    xInt = uint16(x);

                    % Ensure bounds are in range
                    if leftBound < 1
                        leftBound = 1;
                    end
                    if rightBound > width
                        rightBound = width;
                    end
                    if topBound < 1
                        topBound = 1;
                    end
                    if bottomBound > height
                        bottomBound = height;
                    end

                    % Get each half of a rectangle around the current pixel we
                    % are inspecting
                    leftRectangle = uint8(tempImg(topBound:yInt, leftBound:xInt));
                    rightRectangle = uint8(tempImg(yInt+1:bottomBound, xInt+1:rightBound));

                    % Ensure we only compare rectangles that are the same size
                    % to avoid biased results (sizing gets weird near edges)
                    if size(leftRectangle) ~= size(rightRectangle)
                        %disp("Uneven Sizing");
                        continue
                    end

                    % Compute Histograms for each half-rectangle with 32 bins + normalize results
                    %
                    % From https://www.mathworks.com/matlabcentral/answers/85447-comparison-of-two-histograms-using-pdist2
                    [leftCounts, leftLocations] = imhist(leftRectangle, 32);
                    leftCounts = leftCounts / size(leftRectangle, 1) / size(leftRectangle, 2);

                    [rightCounts, rightLocations] = imhist(rightRectangle, 32);
                    rightCounts = rightCounts / size(rightRectangle, 1) / size(rightRectangle, 2);

                    % Compute Chi-Square distance of each histogram
                    % From https://web.archive.org/web/20190131024241/http://www.cs.columbia.edu:80/~mmerler/project/code/pdist2.m 
                    % Equation 9 from the paper
                    dist = pdistNew(leftCounts, rightCounts, 'chisq');
                    rotatedResults(y, x, i, j) = dist;

                end
            end

            % Apply 2nd-order Savitzky-Golay Filtering to results to
            % "enhance local minima and smooth detection peaks"
            
            rotatedResults(:, :, i, j) = savitzkyGolay2D_rle_coupling(height, width, rotatedResults(:, :, i, j), featureScales(i) + 1, featureScales(i) + 1, 2);
            
            % Ensure window size is odd because we need that shit
            if mod(radius, 2) == 0
                radius = radius + 1;
            end
            
            % Actually apply filter
            %rotatedResults(:, :, i, j) = sgolayfilt(rotatedResults(:, :, i, j), 2, radius);
            %imshow(rotatedResults(:, :, i, j), [1, 32]);
            %x = input('x' ,'s');

        end
    end

    %% Rotate results of Filtered Histrogram Distances back into place
    % Takes the computations applied on our pre-rotated texture maps and
    % rotates them back into our original coordinate space
    results = zeros([height, width, numScales + 1, numAngles]);
    for i = 1:numScales + 1
        for j = 1:numAngles
            results(:, :, i, j) = imrotate(rotatedResults(:, :, i, j), rad2deg(angles(j)), 'crop');
        end
    end
    
    
    % See if we should save our results for future use
    if saveRun
        save(matlabFilename, 'results');
    end
    
end  % End Feature Creation

% If we are just loading a previous result, jump to here
if loadRun
    load(matlabFilename)
end

%% Start Building Final Image
% Now that we have mPb(x, y, theta, scale), We need to work on collapsing
% it to mPb(x, y)

       
% Load results into "results" variable
%load(matlabFilename);

% Load Previously Calculated Ground Truths
load('./results/matlab-files/allGroundTruths.mat');

% Load previously trained weights
load('./results/matlab-files/bestGlobalWeightsgPb.mat');

% Will be used to hold mPb(x, y, theta) from paper
newResults = zeros([size(results, 1), size(results, 2), 8]);

% Calculate weighted gradients (mPb(x, y, theta) from paper)
%
% Sums each orientation we have at each scale so we're left
% with a 3D matrix where the first two are the image x/y, and the
% third dimension is the summed orientation across our scales.
counter = 1;
for s = 1:4
   for o = 1:8
       newResults(:, :, o) = newResults(:, :, o) + bestGlobalWeights(1, counter) * results(:, :, s, o);
       counter = counter + 1;
   end
end

% Transform mPb(x, y, theta) to mPb(x, y) by taking max at each pixel along
% the orientation dimension
mPbResults = mPb(newResults);
size1 = size(mPbResults, 1);
size2 = size(mPbResults, 2);

% Take mPb(x, y) image, add a slight border since we get some noise
% Normalize + Add 15px Borders + Invert to match our ground truths
pbNorm = mat2gray(mPbResults);
bounds = find(pbNorm < 0.25);
pbNorm(bounds) = 1;
pbNorm(1:17, :) = 1;
pbNorm(:, 1:17) = 1;
pbNorm(size1-17:size1, :) = 1;
pbNorm(:, size2-17:size2) = 1;
pbNorm = imcomplement(pbNorm);
factor = 255 / max(pbNorm(:));
displayImg = uint8(pbNorm .* factor);

% Display original image vs mPb image
%imshow(pbNorm);
imshow([img, displayImg]);
title("Original Image vs mPb Image");
x = input('Showing Original Image vs mPb(x, y) Image', 's');

% Show ground truth edges against calculated edges
%imshow([truths{1}, pbNorm]);
%x = input("Showing Ground Truth Segmentation vs ours", 's');

% Display contoured mPb image
imshow(displayImg);
title("mPb Image with Edge Intensity Highlighted");
colormap jet
colorbar
x = input('Displaying mPb Image with Edge Intensity Highlighted', 's');