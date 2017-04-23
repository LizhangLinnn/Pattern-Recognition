load wine.data.csv

% Split the data into 3 sets for Training (118), Validation (20), 
% and Testing (42). Move the class identifiers to separate vectors
loop = 500;
accuracy_Chi2 = zeros(loop,1);
accuracy_Chi2_norm = zeros(loop,1);
accuracy_Corr = zeros(loop,1);
accuracy_Corr_norm = zeros(loop,1);
accuracy_hist = zeros(loop,1);
accuracy_hist_norm = zeros(loop,1);
accuracy_L1 = zeros(loop,1);
accuracy_L2 = zeros(loop,1);
accuracy_L1_norm = zeros(loop,1);
accuracy_L2_norm = zeros(loop,1);
accuracy_Maha = zeros(loop,1);
accuracy_Maha1 = zeros(loop,1);
accuracy_Maha2 = zeros(loop,1);
accuracy_Maha3 = zeros(loop,1);
accuracy_Maha_norm = zeros(loop,1);
accuracy_Maha_norm1 = zeros(loop,1);
accuracy_Maha_norm2 = zeros(loop,1);
accuracy_Maha_norm3 = zeros(loop,1);

for i=1:loop

data = wine_data(:,2:14);
classifier = wine_data(:,1);


ClassOneRand = randsample(1:59,59);
ClassTwoRand = randsample(60:130,71);
ClassThreeRand = randsample(131:178,48);
%118 training data
TrainingData = wine_data(vertcat(ClassOneRand(1:39)', ClassTwoRand(1:47)', ClassThreeRand(1:32)'),:);
%20 validation data
ValidationData = wine_data(vertcat(ClassOneRand(40:46)', ClassTwoRand(48:55)', ClassThreeRand(33:37)'),:);
%40 test data
TestData = wine_data(vertcat(ClassOneRand(47:59)', ClassTwoRand(56:71)', ClassThreeRand(38:48)'),:);
train_validate = vertcat(TrainingData,ValidationData);
test = TestData;


train_validate_norm = normc(wine_data(:,2:14));
train_validate_norm = horzcat(train_validate(:,1),vertcat(train_validate_norm([ClassOneRand(1:39)...
    ,ClassTwoRand(1:47),ClassThreeRand(1:32)],:),(train_validate_norm([ClassOneRand(40:46)...
    ,ClassTwoRand(48:55),ClassThreeRand(33:37)],:))));
test_norm = normc(wine_data(:,2:14));
test_norm = horzcat(test(:,1),test_norm([ClassOneRand(47:59),ClassTwoRand(56:71),ClassThreeRand(38:48)],:));


% estimate covariance matrix (All features from all classes)
% A.a
cov_All = cov(train_validate(:,2:14));
mean_All = mean(train_validate(:,2:14));

% A.b
cov_All_norm = cov(train_validate_norm(:,2:14));
mean_All_norm = mean(train_validate_norm(:,2:14));

% estimate covariance matrix (independently from class 1, 2 and 3)
% A.a
idx_ClassOne = horzcat((1:39),(119:125));
cov_ClassOne = cov(train_validate(idx_ClassOne,2:14));
mean_ClassOne = mean(train_validate(idx_ClassOne,2:14));

idx_ClassTwo = horzcat((40:86),(126:133));
cov_ClassTwo = cov(train_validate(idx_ClassTwo,2:14));
mean_ClassTwo = mean(train_validate(idx_ClassTwo,2:14));

idx_ClassThree = horzcat((87:118),(134:138));
cov_ClassThree = cov(train_validate(idx_ClassThree,2:14));
% cov_ClassThree = cov(wine_data(131:end,2:14));
mean_ClassThree = mean(train_validate(idx_ClassThree,2:14));

% A.b
cov_ClassOne_norm = cov(train_validate_norm(idx_ClassOne,2:14));
mean_ClassOne_norm = mean(train_validate_norm(idx_ClassOne,2:14));

cov_ClassTwo_norm = cov(train_validate_norm(idx_ClassTwo,2:14));
mean_ClassTwo_norm = mean(train_validate_norm(idx_ClassTwo,2:14));

cov_ClassThree_norm = cov(train_validate_norm(idx_ClassThree,2:14));
mean_ClassThree_norm = mean(train_validate_norm(idx_ClassThree,2:14));



% L1 - original

% Train model
[L1, validate_L1] = trainClassifier_L1(train_validate(1:118,:));
yfit_L1 = L1.predictFcn(test(:,2:14));
result = yfit_L1 - test(:,1);


% Test with testing data
accuracy_L1(i) = 1-(nnz(yfit_L1 - test(:,1))/size(yfit_L1 - test(:,1),1));
% L1 - normalised

% Train model
[L1_norm, validate_L1_norm] = trainClassifier_L1(train_validate_norm(1:118,:));
yfit_L1_norm = L1_norm.predictFcn(test_norm(:,2:14));
result = yfit_L1_norm - test_norm(:,1);

% Test with testing data
accuracy_L1_norm(i) = 1-(nnz(yfit_L1_norm - test_norm(:,1))/size(yfit_L1_norm - test_norm(:,1),1));

% L2

% Train model
[L2, validate_L2] = trainClassifier_L2_new(train_validate(1:118,:));

yfit_L2 = L2.predictFcn(test(:,2:14));
result = yfit_L2 - test(:,1);

accuracy_L2(i) = 1-(nnz(yfit_L2 - test(:,1))/size(yfit_L2 - test(:,1),1));
% L2 - normalised

% Train model
[L2_norm, validate_L2_norm] = trainClassifier_L2_new(train_validate_norm(1:118,:));

yfit_L2_norm = L2_norm.predictFcn(test_norm(:,2:14));
result = yfit_L2_norm - test_norm(:,1);

accuracy_L2_norm(i)= 1-(nnz(yfit_L2_norm - test_norm(:,1))/size(yfit_L2_norm - test_norm(:,1),1));

% Chi2
[accuracy_Chi2(i), ~] = Chi2(train_validate(1:118,:),test);
% Chi2 - normalised
[accuracy_Chi2_norm(i), ~] = Chi2(train_validate_norm(1:118,:),test_norm);

% Histogram
[accuracy_hist(i),~] = histogram_intersection_norm(train_validate(1:118,:),test);

% Histogram normalised
% [accuracy_hist_norm] = histogram_intersection(train_validate_norm(1:118,:),test_norm);
[accuracy_hist_norm(i),~] = histogram_intersection_norm(train_validate_norm(1:118,:),test_norm);

% Correlation

% Train model
[Corr, validate_Corr] = trainClassifier_Corr(train_validate(1:118,:));

yfit_Corr = Corr.predictFcn(test(:,2:14));
accuracy_Corr(i) = 1-(nnz(yfit_Corr - test(:,1))/size(yfit_Corr - test(:,1),1));

% Correlation - normalised

% Train model
[Corr_norm, validate_Corr_norm] = trainClassifier_Corr(train_validate_norm(1:118,:));

yfit_Corr_norm = Corr_norm.predictFcn(test_norm(:,2:14));
accuracy_Corr_norm(i) = 1-(nnz(yfit_Corr_norm - test(:,1))/size(yfit_Corr_norm - test(:,1),1));
% Mahalanobis

% Train model
[Maha, validate_Maha] = trainClassifier_Maha(train_validate(1:118,:),cov_All);

yfit_Maha = Maha.predictFcn(test(:,2:14));
accuracy_Maha(i) = 1-(nnz(yfit_Maha - test(:,1))/size(yfit_Maha - test(:,1),1));
%Mahalanobis Class 1

% Train model
[Maha, validate_Maha] = trainClassifier_Maha(train_validate(1:118,:),cov_ClassOne);

yfit_Maha = Maha.predictFcn(test(:,2:14));
accuracy_Maha1(i) = 1-(nnz(yfit_Maha - test(:,1))/size(yfit_Maha - test(:,1),1));
% Mahalanobis Class 2

% Train model
[Maha, validate_Maha] = trainClassifier_Maha(train_validate(1:118,:),cov_ClassTwo);

yfit_Maha = Maha.predictFcn(test(:,2:14));
accuracy_Maha2(i) = 1-(nnz(yfit_Maha - test(:,1))/size(yfit_Maha - test(:,1),1));

% Mahalanobis Class 3

% Train model
[Maha, validate_Maha] = trainClassifier_Maha(train_validate(1:118,:),cov_ClassThree);

yfit_Maha = Maha.predictFcn(test(:,2:14));
accuracy_Maha3(i) = 1-(nnz(yfit_Maha - test(:,1))/size(yfit_Maha - test(:,1),1));

% Mahalanobis - normalised

% Train model
[Maha_norm, validate_Maha_norm] = trainClassifier_Maha(train_validate_norm(1:118,:),cov_All_norm);

yfit_Maha_norm = Maha_norm.predictFcn(test_norm(:,2:14));
accuracy_Maha_norm(i) = 1-(nnz(yfit_Maha_norm - test(:,1))/size(yfit_Maha_norm - test(:,1),1));
% Mahalanobis - normalised class1

% Train model
[Maha_norm, validate_Maha_norm] = trainClassifier_Maha(train_validate_norm(1:118,:),cov_ClassOne_norm);

yfit_Maha_norm = Maha_norm.predictFcn(test_norm(:,2:14));
accuracy_Maha_norm1(i) = 1-(nnz(yfit_Maha_norm - test(:,1))/size(yfit_Maha_norm - test(:,1),1));

% Mahalanobis - normalised class2

% Train model
[Maha_norm, validate_Maha_norm] = trainClassifier_Maha(train_validate_norm(1:118,:),cov_ClassTwo_norm);

yfit_Maha_norm = Maha_norm.predictFcn(test_norm(:,2:14));
accuracy_Maha_norm2(i) = 1-(nnz(yfit_Maha_norm - test(:,1))/size(yfit_Maha_norm - test(:,1),1));

% Mahalanobis - normalised class3

% Train model
[Maha_norm, validate_Maha_norm] = trainClassifier_Maha(train_validate_norm(1:118,:),cov_ClassThree_norm);

yfit_Maha_norm = Maha_norm.predictFcn(test_norm(:,2:14));
accuracy_Maha_norm3(i) = 1-(nnz(yfit_Maha_norm - test(:,1))/size(yfit_Maha_norm - test(:,1),1));


end
accuracy_Chi2 = mean(accuracy_Chi2);
accuracy_Chi2_norm = mean(accuracy_Chi2_norm);
accuracy_Corr = mean(accuracy_Corr);
accuracy_Corr_norm = mean(accuracy_Corr_norm);
accuracy_hist = mean(accuracy_hist);
accuracy_hist_norm = mean(accuracy_hist_norm);
accuracy_L1 = mean(accuracy_L1);
accuracy_L2 = mean(accuracy_L2);
accuracy_L1_norm = mean(accuracy_L1_norm);
accuracy_L2_norm = mean(accuracy_L2_norm);
accuracy_Maha = mean(accuracy_Maha);
accuracy_Maha1 = mean(accuracy_Maha1);
accuracy_Maha2 = mean(accuracy_Maha2);
accuracy_Maha3 = mean(accuracy_Maha3);
accuracy_Maha_norm = mean(accuracy_Maha_norm);
accuracy_Maha_norm1 = mean(accuracy_Maha_norm1);
accuracy_Maha_norm2 = mean(accuracy_Maha_norm2);
accuracy_Maha_norm3 = mean(accuracy_Maha_norm3);

%%
subplot(121);
name = {'L2';...
    'L1';...
    'Chi2';...
    'Hist';...
    'Corr';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:9]',100*[1-0.7490;1-0.8172;1-0.9285;1-0.6510;1-0.8353;1-0.9078;1-0.9722;1-0.9762;1-0.9602]);
set(gca,'xticklabel',name);
title('raw data - error rate of NN classification with different metrics');
xlabel('metrics');
ylabel('error rate / %');
%
subplot(122);
name = {'L2';...
    'L1';...
    'Chi2';...
    'Hist';...
    'Corr';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:9]',100*[1-0.9570;1-0.9633;1-0.9687;1-0.7028;1-0.9523;1-0.9078;1-0.9722;1-0.9762;1-0.9602]);
set(gca,'xticklabel',name);
title('normalised data - error rate of NN classification with different metrics');
xlabel('metrics');
ylabel('error rate / %');
