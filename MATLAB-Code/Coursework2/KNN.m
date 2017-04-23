%% Initialisation

clear; clc

load class_label.mat
load wine.mat
load all.mat
load train_validate.mat
load test.mat
%% Normalise training data

train_validate_norm = normr(train_validate(:,2:14));
train_validate_norm = horzcat(train_validate(:,1),train_validate_norm);
test_norm = normr(test(:,2:14));
test_norm = horzcat(test(:,1),test_norm);
%% L1 - original

% Train model
[L1, validate_L1] = trainClassifier_L1(train_validate);

% Test with testing data
yfit_L1 = L1.predictFcn(test(:,2:14));
accuracy_L1 = 1-(nnz(yfit_L1 - test(:,1))/size(yfit_L1 - test(:,1),1));
%% L1 - normalised

% Train model
[L1_norm, validate_L1_norm] = trainClassifier_L1(train_validate_norm);

% Test with testing data
yfit_L1_norm = L1_norm.predictFcn(test_norm(:,2:14));
accuracy_L1_norm = 1-(nnz(yfit_L1_norm - test(:,1))/size(yfit_L1_norm - test(:,1),1));

%% L2

% Train model
[L2, validate_L2] = trainClassifier_L2(train_validate);

yfit_L2 = L2.predictFcn(test(:,2:14));
accuracy_L2= 1-(nnz(yfit_L2 - test(:,1))/size(yfit_L2 - test(:,1),1));
%% L2 - normalised

% Train model
[L2_norm, validate_L2_norm] = trainClassifier_L2(train_validate_norm);

yfit_L2_norm = L2_norm.predictFcn(test_norm(:,2:14));
accuracy_L2_norm= 1-(nnz(yfit_L2_norm - test(:,1))/size(yfit_L2_norm - test(:,1),1));

%% Chi2
[accuracy_Chi2] = Chi2(train_validate(1:118,:),test);
%% Chi2 - normalised
[accuracy_Chi2_norm] = Chi2(train_validate_norm(1:118,:),test_norm);

%% Histogram
%% Correlation

% Train model
[Corr, validate_Corr] = trainClassifier_Corr(train_validate);

yfit_Corr = Corr.predictFcn(test(:,2:14));
accuracy_Corr = 1-(nnz(yfit_Corr - test(:,1))/size(yfit_Corr - test(:,1),1));

%% Correlation - normalised

% Train model
[Corr_norm, validate_Corr_norm] = trainClassifier_Corr(train_validate_norm);

yfit_Corr_norm = Corr_norm.predictFcn(test_norm(:,2:14));
accuracy_Corr_norm = 1-(nnz(yfit_Corr_norm - test(:,1))/size(yfit_Corr_norm - test(:,1),1));
%% Mahalanobis

% Train model
[Maha, validate_Maha] = trainClassifier_Maha(train_validate);

yfit_Maha = Maha.predictFcn(test(:,2:14));
accuracy_Maha = 1-(nnz(yfit_Maha - test(:,1))/size(yfit_Maha - test(:,1),1));

%% Mahalanobis - normalised

% Train model
[Maha_norm, validate_Maha_norm] = trainClassifier_Maha(train_validate_norm);

yfit_Maha_norm = Maha_norm.predictFcn(test_norm(:,2:14));
accuracy_Maha_norm = 1-(nnz(yfit_Maha_norm - test(:,1))/size(yfit_Maha_norm - test(:,1),1));
