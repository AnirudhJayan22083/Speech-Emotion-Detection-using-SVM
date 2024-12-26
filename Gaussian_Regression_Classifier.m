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
        labels = [labels; [1 0]];
    elseif contains(name, 'SAD')
        [audioData, fs] = audioread(fullfile(dataFolder, audioFiles(i).name));
        coeffs = mfcc(audioData, fs);
        meanCoeffs = mean(coeffs, 1);
        features = [features; meanCoeffs];
        labels = [labels; [0 1]];
    end
end



% Normalize training features
featurest=features(1:round(0.7*length(features)),:)
labelst = labels(1:round(0.7*length(features)),:)


% Test features and labels extraction
features_test = features(round(0.7*length(features)):length(features),:)
labels_test = labels(round(0.7*length(features)):length(features),:)

[U, S, V] = svd(featurest, 'econ');
k = 2; % Increased number of dimensions to retain more information
U_reduced = U(:, 1:k);
S_reduced = S(1:k, 1:k);
V_reduced = V(:, 1:k);

% Reduced training features
reduced_features = U_reduced * S_reduced;

reduced_features_test = features_test * V_reduced;

B = rand(k,5000);
M = reduced_features*B;
M=cos(M);
w = pinv(M)*labelst;

Mt=reduced_features_test*B;
Mt=cos(Mt);
yp= round(Mt*w)

test_accuracy = sum(yp == labels_test) / length(labels_test) * 100

