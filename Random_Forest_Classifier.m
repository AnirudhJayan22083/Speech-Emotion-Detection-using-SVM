clc; clear all; close all;

% Specify the folder containing the audio files
dataFolder = 'C:\Users\aniru\OneDrive\Documents\MFC\emotion_dataset'; % Update with your path
audioFiles = dir(fullfile(dataFolder, '*.wav')); % Adjust file type if needed

% Initialize variables for features and labels
features = [];
labels = [];

% Extract features and labels from audio files (training set)
for i = 1:round(length(audioFiles))
    [~, name, ~] = fileparts(audioFiles(i).name);
    if contains(name, 'HAP')
        [audioData, fs] = audioread(fullfile(dataFolder, audioFiles(i).name));
        coeffs = mfcc(audioData, fs);
        meanCoeffs = mean(coeffs, 1);
        features = [features; meanCoeffs];
        labels = [labels; 1]; % Label 1 for "Happy"
    elseif contains(name, 'SAD')
        [audioData, fs] = audioread(fullfile(dataFolder, audioFiles(i).name));
        coeffs = mfcc(audioData, fs);
        meanCoeffs = mean(coeffs, 1);
        features = [features; meanCoeffs];
        labels = [labels; 0]; % Label 0 for "Sad"
    end
end

% Split into training and test sets
trainIdx = 1:round(0.7 * length(features));
testIdx = round(0.7 * length(features)) + 1:length(features);

features_train = features(trainIdx, :);
labels_train = labels(trainIdx, :);
features_test = features(testIdx, :);
labels_test = labels(testIdx, :);

% Perform SVD for dimensionality reduction
[U, S, V] = svd(features_train, 'econ');
k = 10; % Number of dimensions to retain
U_reduced = U(:, 1:k);
S_reduced = S(1:k, 1:k);
V_reduced = V(:, 1:k);

% Reduced training features
reduced_features_train = U_reduced * S_reduced;

% Reduced test features
reduced_features_test = features_test * V_reduced;

% Train a Random Forest classifier using TreeBagger
numTrees = 100; % Number of trees in the Random Forest
randomForestModel = TreeBagger(numTrees, reduced_features_train, labels_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'NumPredictorsToSample', 'all', ...
    'MinLeafSize', 1);

% Display OOB error
figure;
oobErrorBagged = oobError(randomForestModel);
plot(oobErrorBagged);
xlabel('Number of Grown Trees');
ylabel('Out-of-Bag Classification Error');
title('OOB Error for Random Forest');

% Predict on the test set
[yp_test, scores] = predict(randomForestModel, reduced_features_test);

% Convert predictions to numeric for comparison
yp_test = str2double(yp_test);

% Calculate test accuracy
test_accuracy = sum(yp_test == labels_test) / length(labels_test) * 100;
fprintf("Test Accuracy: %.2f%%\n", test_accuracy);

% Confusion matrix
confMat = confusionmat(labels_test, yp_test);

% Display the confusion matrix
disp('Confusion Matrix:');
disp(confMat);

% Plot the confusion matrix
figure;
confusionchart(confMat, {'Sad', 'Happy'});
title('Confusion Matrix for Speech Emotion Detection');

