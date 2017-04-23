
load wine.data.csv

% Split the data into 3 sets for Training (118), Validation (20), 
% and Testing (42). Move the class identifiers to separate vectors
loop = 100;

accuracy_kmeans_sqeuclidean =zeros(loop,1);
accuracy_kmeans_cityblock =zeros(loop,1);
accuracy_kmeans_correlation =zeros(loop,1);
accuracy_kmeans_cosine =zeros(loop,1);
accuracy_kmeans_maha = zeros(loop,1);
accuracy_kmeans_maha1 = zeros(loop,1);
accuracy_kmeans_maha2 = zeros(loop,1);
accuracy_kmeans_maha3 = zeros(loop,1);


for n=1:loop

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

%% normalised
% 'sqeuclidean'
k = 3;
% avoid local minima
[idx_sqeuclidean,cent_sqeuclidean,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','sqeuclidean','Display','final','Replicates',100);
sum(sumdist);


% convert test data indexes to the label we assigned in this case
TestData_norm_convert = test_norm;
for i=1:1:size(TestData_norm_convert,1)
    if(TestData_norm_convert(i,1) == 1)
        TestData_norm_convert(i,1) = mode(idx_sqeuclidean(1:39));
    elseif(TestData_norm_convert(i,1) == 3)
        TestData_norm_convert(i,1) = mode(idx_sqeuclidean(87:118));
    else
        TestData_norm_convert(i,1) = mode(idx_sqeuclidean(40:86));
    end
end

% input test data
d_sqeuclidean = zeros(size(test_norm,1),size(cent_sqeuclidean,1));
min_d = zeros(size(test_norm,1),1);
PredictedClass = zeros(size(test_norm,1),1);
CorrectCount = 0;
PredictedClass = knnsearch(cent_sqeuclidean,TestData_norm_convert(:,2:14),'k',1,'Distance','euclidean'); 

for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_sqeuclidean,1)
%         for dim=2:1:size(test_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_sqeuclidean(i,j) = (TestData_norm_convert(i,dim)-cent_sqeuclidean(j,dim-1))^2 + d_sqeuclidean(i);
%         end
%     end
%     % return the label of the mapped data point (PredictedClass)
%     [min_d(i) PredictedClass(i)] = min(d_sqeuclidean(i,:));

    %     PredictedClass = knnsearch(cent_sqeuclidean,TestData_norm_convert(:,2:14),'Distance','euclidean');
    if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_sqeuclidean(n) = CorrectCount/size(test_norm,1);




%%
% cityblock
% avoid local minima
k = 3;
[idx_cityblock,cent_cityblock,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','cityblock','Display','final','Replicates',100);
sum(sumdist);


% convert test data indexes to the label we assigned in this case
TestData_norm_convert = test_norm;
for i=1:1:size(TestData_norm_convert,1)
    if(TestData_norm_convert(i,1) == 1)
        TestData_norm_convert(i,1) = mode(idx_cityblock(1:39));
    elseif(TestData_norm_convert(i,1) == 3)
        TestData_norm_convert(i,1) = mode(idx_cityblock(87:118));
    else
        TestData_norm_convert(i,1) = mode(idx_cityblock(40:86));   
    end
end

% input test data
CorrectCount = 0;
PredictedClass = knnsearch(cent_cityblock,TestData_norm_convert(:,2:14),'k',1,'Distance','cityblock');
for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
    if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_cityblock(n) = CorrectCount/size(test_norm,1);


%% normalised
% 'cosine'
% avoid local minima
k = 3;
[idx_cosine,cent_cosine,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','cosine','Display','final','Replicates',100);
sum(sumdist);


% convert test data indexes to the label we assigned in this case
TestData_norm_convert = test_norm;
for i=1:1:size(TestData_norm_convert,1)
    if(TestData_norm_convert(i,1) == 1)
        TestData_norm_convert(i,1) = mode(idx_cosine(1:39));
    else if(TestData_norm_convert(i,1) == 3)
        TestData_norm_convert(i,1) = mode(idx_cosine(87:118));
        else
        TestData_norm_convert(i,1) = mode(idx_cosine(40:86));
        end
    end
end

% input test data
min_d = zeros(size(test_norm,1),1);
CorrectCount = 0;
PredictedClass = knnsearch(cent_cosine,TestData_norm_convert(:,2:14),'k',1,'Distance','cosine');
    
for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
    if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_cosine(n) = CorrectCount/size(test_norm,1);




%% normalised
% 'correlation' . best:0.7750, worst 0.65
% avoid local minima
k = 3;
[idx_correlation,cent_correlation,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','correlation','Display','final','Replicates',100);
sum(sumdist);


% convert test data indexes to the label we assigned in this case
TestData_norm_convert = test_norm;
for i=1:1:size(TestData_norm_convert,1)
    if(TestData_norm_convert(i,1) == 1)
        TestData_norm_convert(i,1) = mode(idx_correlation(1:39));
    else if(TestData_norm_convert(i,1) == 3)
        TestData_norm_convert(i,1) = mode(idx_correlation(87:118));
        else
        TestData_norm_convert(i,1) = mode(idx_correlation(40:86));
        end
    end
end

% input test data
min_d = zeros(size(test_norm,1),1);
CorrectCount = 0;
PredictedClass = knnsearch(cent_correlation,TestData_norm_convert(:,2:14),'k',1,'Distance','correlation');
for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
    if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_correlation(n) = CorrectCount/size(test_norm,1);




%% normalised
% 'Mahalanobis' .
% avoid local minima
k = 3;
[eig_vec, eig_val] = eig(cov_All_norm);
G = (((eig_val).^0.5)' * eig_vec');

train_validate_norm_maha = train_validate_norm(1:118,2:14)*G;

[idx_maha,cent_maha,sumdist] = kmeans(train_validate_norm_maha, k, 'Distance','sqeuclidean','Display','final','Replicates',100);
sum(sumdist);

% convert test data indexes to the label we assigned in this case
TestData_norm_convert = horzcat(test_norm(:,1),test_norm(:,2:14)*G);
for i=1:1:size(TestData_norm_convert,1)
    if(TestData_norm_convert(i,1) == 1)
        TestData_norm_convert(i,1) = mode(idx_maha(1:39));
    elseif(TestData_norm_convert(i,1) == 3)
        TestData_norm_convert(i,1) = mode(idx_maha(87:118));
    else
        TestData_norm_convert(i,1) = mode(idx_maha(40:86));
    end
end

% input test data
CorrectCount = 0;
PredictedClass = knnsearch(cent_maha,TestData_norm_convert(:,2:14),'k',1,'Distance','euclidean');
for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
        if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_maha(n) = CorrectCount/size(test_norm,1);

%% normalised
% 'Mahalanobis CLass one cov' .
% avoid local minima
k = 3;
[eig_vec, eig_val] = eig(cov_ClassOne_norm);
G = (((eig_val).^0.5)' * eig_vec');

train_validate_norm_maha = train_validate_norm(1:118,2:14)*G;

[idx_maha,cent_maha,sumdist] = kmeans(train_validate_norm_maha, k, 'Distance','sqeuclidean','Display','final','Replicates',100);
sum(sumdist);

% convert test data indexes to the label we assigned in this case
TestData_norm_convert = horzcat(test_norm(:,1),test_norm(:,2:14)*G);
for i=1:1:size(TestData_norm_convert,1)
    if(TestData_norm_convert(i,1) == 1)
        TestData_norm_convert(i,1) = mode(idx_maha(1:39));
    elseif(TestData_norm_convert(i,1) == 3)
        TestData_norm_convert(i,1) = mode(idx_maha(87:118));
    else
        TestData_norm_convert(i,1) = mode(idx_maha(40:86));
    end
end

% input test data
min_d = zeros(size(test_norm,1),1);
CorrectCount = 0;
PredictedClass = knnsearch(cent_maha,TestData_norm_convert(:,2:14),'k',1,'Distance','euclidean');
for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
        if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_maha1(n)  = CorrectCount/size(test_norm,1);

%% normalised
% 'Mahalanobis' .
% avoid local minima
k = 3;
[eig_vec, eig_val] = eig(cov_ClassTwo_norm);
G = (((eig_val).^0.5)' * eig_vec');

train_validate_norm_maha = train_validate_norm(1:118,2:14)*G;

[idx_maha,cent_maha,sumdist] = kmeans(train_validate_norm_maha, k, 'Distance','sqeuclidean','Display','final','Replicates',100);
sum(sumdist);

% convert test data indexes to the label we assigned in this case
TestData_norm_convert = horzcat(test_norm(:,1),test_norm(:,2:14)*G);
for i=1:1:size(TestData_norm_convert,1)
    if(TestData_norm_convert(i,1) == 1)
        TestData_norm_convert(i,1) = mode(idx_maha(1:39));
    elseif(TestData_norm_convert(i,1) == 3)
        TestData_norm_convert(i,1) = mode(idx_maha(87:118));
    else
        TestData_norm_convert(i,1) = mode(idx_maha(40:86));
    end
end

% input test data
min_d = zeros(size(test_norm,1),1);
CorrectCount = 0;
PredictedClass = knnsearch(cent_maha,TestData_norm_convert(:,2:14),'k',1,'Distance','euclidean');
for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
        if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_maha2(n)  = CorrectCount/size(test_norm,1);

%% normalised
% 'Mahalanobis' .
% avoid local minima
k = 3;
[eig_vec, eig_val] = eig(cov_ClassThree_norm);
G = (((eig_val).^0.5)' * eig_vec');

train_validate_norm_maha = train_validate_norm(1:118,2:14)*G;

[idx_maha,cent_maha,sumdist] = kmeans(train_validate_norm_maha, k, 'Distance','sqeuclidean','Display','final','Replicates',100);
sum(sumdist);

% convert test data indexes to the label we assigned in this case
TestData_norm_convert = horzcat(test_norm(:,1),test_norm(:,2:14)*G);
for i=1:1:size(TestData_norm_convert,1)
    if(TestData_norm_convert(i,1) == 1)
        TestData_norm_convert(i,1) = mode(idx_maha(1:39));
    elseif(TestData_norm_convert(i,1) == 3)
        TestData_norm_convert(i,1) = mode(idx_maha(87:118));
    else
        TestData_norm_convert(i,1) = mode(idx_maha(40:86));
    end
end

% input test data
min_d = zeros(size(test_norm,1),1);
CorrectCount = 0;
PredictedClass = knnsearch(cent_maha,TestData_norm_convert(:,2:14),'k',1,'Distance','euclidean');
for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
        if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_maha3(n)  = CorrectCount/size(test_norm,1);

end

accuracy_kmeans_sqeuclidean =mean(accuracy_kmeans_sqeuclidean);
accuracy_kmeans_cityblock =mean(accuracy_kmeans_cityblock);
accuracy_kmeans_correlation =mean(accuracy_kmeans_correlation);
accuracy_kmeans_cosine =mean(accuracy_kmeans_cosine);
accuracy_kmeans_maha = mean(accuracy_kmeans_maha);
accuracy_kmeans_maha1 = mean(accuracy_kmeans_maha1);
accuracy_kmeans_maha2 = mean(accuracy_kmeans_maha2);
accuracy_kmeans_maha3 = mean(accuracy_kmeans_maha3);


%%
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9198;1-0.9365;1-0.8903;1-0.8925;1-0.9245;1-0.9335;1-0.9335;1-0.9240]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=3)');
xlabel('metrics');
ylabel('error rate / %');