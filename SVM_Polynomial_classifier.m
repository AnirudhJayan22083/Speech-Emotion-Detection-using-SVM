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
        labels = [labels; 1];
    elseif contains(name, 'SAD')
        [audioData, fs] = audioread(fullfile(dataFolder, audioFiles(i).name));
        coeffs = mfcc(audioData, fs);
        meanCoeffs = mean(coeffs, 1);
        features = [features; meanCoeffs];
        labels = [labels; 0];
    end
end



% Normalize training features
featurest=features(1:round(0.7*length(features)),:)
labelst = labels(1:round(0.7*length(features)),:)


% Test features and labels extraction
features_test = features(round(0.7*length(features)):length(features),:)
labels_test = labels(round(0.7*length(features)):length(features),:)

[U, S, V] = svd(features, 'econ');
k = 10; % Increased number of dimensions to retain more information
U_reduced = U(:, 1:k);
S_reduced = S(1:k, 1:k);
V_reduced = V(:, 1:k);

% Reduced training features
reduced_features = U_reduced * S_reduced;

reduced_features_test = features_test * V_reduced;


% SVM with Hyperparameter Tuning
C_values = [0.1, 1, 10, 100]; % BoxConstraint values to test
bestAccuracy = 0;
bestModel = [];

for C = C_values
    % Train SVM
    SVM_model = fitcsvm(reduced_features, labels, ...
        'KernelFunction', 'polynomial', ...
        'BoxConstraint', C, ...
        'Standardize', true);

    % Cross-validation
    cvSVMModel = crossval(SVM_model, 'KFold', 5);
    cvAccuracy = 1 - kfoldLoss(cvSVMModel, 'LossFun', 'classiferror');
    fprintf("C = %.2f, Cross-Validation Accuracy: %.2f%%\n", C, cvAccuracy * 100);

    % Store the best model
    if cvAccuracy > bestAccuracy
        bestAccuracy = cvAccuracy;
        bestModel = SVM_model;
    end
end

% Use the best model for testing
yp_test = predict(bestModel, reduced_features_test);

% Calculate test accuracy
test_accuracy = sum(yp_test == labels_test) / length(labels_test) * 100;
fprintf("Best Test Accuracy: %.2f%%\n", test_accuracy);

confMat = confusionmat(labels_test, yp_test);

% Display the confusion matrix
disp('Confusion Matrix:');
disp(confMat);

% Plot the confusion matrix
figure;
confusionchart(confMat, {'Sad', 'Happy'});
title('Confusion Matrix for Speech Emotion Detection');



