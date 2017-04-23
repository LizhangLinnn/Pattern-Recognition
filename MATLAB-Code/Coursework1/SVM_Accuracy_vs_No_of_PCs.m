clc
clear
close all

%load face data
load face.mat

%% Q1

%10-fold crossvalidation
%10 items in each class and 9 data into training set, 1 into test set
k=10;                               %Define ratio of partition, k is the proportion sorted into test set
c = cvpartition(l,'Kfold',k);       %Create partition object



%% NN classification method

accuracy = zeros(10,468);
elapsedTime = zeros(10,468);
for j=1:10

TestIdx=find(c.test(j));                    %Create index list for test set
TrainingIdx=find(c.training(j));            %Index list for training set
testData=X(:,TestIdx);              
trainData=X(:,TrainingIdx);

%find mean face image for training data
mean_face = mean(trainData,2); %return a column vector which is the mean of training data
phi_train = trainData - repmat(mean_face,[1,468]); %Obtain train data
S = (phi_train' * phi_train)/size(phi_train,2); %A'A

%compute and normalise the eigenvectors of covariance matrix S
[eig_vec, eig_val] = eig(S);
eig_vec = phi_train * eig_vec;
eig_vec = normc(eig_vec);
[eig_val_sort, eig_val_sort_index] = sort(diag(eig_val),'descend');
% 
% 
    for M = 1:size(trainData,2)-250
        %compute the best M eigenvectors
        tic;
        M_eig_vec = eig_vec(:, eig_val_sort_index(1:M));
            train_projection = phi_train' * M_eig_vec;

            test_projection = testData'* M_eig_vec;

            train = train_projection;
            test = test_projection;
               
        decision_val = zeros(52,52);
        
        for i = 1:52
        % creating different labels for each loop 
            label_test = -ones(52,1);
            label_test(i) = 1;
            label_train = -ones(468,1);
            label_train(((i-1)*9+1):(i*9)) = 1;


        svm_1vAll = fitcsvm(...
        train, ...
        label_train, ...
        'KernelFunction', 'rbf', ...
        'PolynomialOrder', [], ...
        'KernelScale', 39, ...
        'BoxConstraint',1, ...
        'Standardize', true, ...
        'ClassNames', [1; -1]);

        [label,score] = predict(svm_1vAll,test);
        decision_val(:,i) = score(:,1);
        end
        [x,result] = max(decision_val);
        accuracy(j,M) = 1 - (nnz(result - (1:52))/52);
        
        elapsedTime(j,M)=toc;
    end      
end
figure;
% plot accuracy (success rate)
[ax, h1, h2] = plotyy([1:468],mean(accuracy)*100,[1:468],mean(elapsedTime));
title('Accuracy against no. of Eigenvectors used as PCA bases');
xlabel('no. of eigenvectors');
ylabel('rate of success / %');
axes(ax(2)); ylabel('Time / second');
set(ax(2),'fontsize',16);
set(ax(1),'fontsize',16);
set(ax(2),'linewidth',1);
set(ax(1),'linewidth',1);
set(h1,'linewidth',2);
set(h2,'linewidth',2);
alldatacursors = findall(gcf,'type','hggroup');
set(alldatacursors,'FontSize',18);
% %% example success and failure cases
% 
% %success
% figure;
% subplot(121);
% imagesc(reshape(TrainData(:,LB_PredictedData(1,1)),56,46));
% title('Predicted Image');
% subplot(122);
% imagesc(reshape(TestData(:,1),56,46));
% title('Actual Image');
% colormap gray
% 
% %fail
% figure;
% subplot(121);
% imagesc(reshape(TrainData(:,LB_PredictedData(2,1)),56,46));
% title('Predicted Image');
% subplot(122);
% imagesc(reshape(TestData(:,2),56,46));
% title('Actual Image');
% colormap gray
% 
% mode
% 
% %% confusion matrix
% % true positive
% % faulse positive
% % predicted data is correct if the image with minimum error lies in the
% LB_PredictedData = ceil(LB_PredictedData/9);
% LB_ActualData = [1:1:52];
% Confusion_Matrix_NN = confusionmat(LB_ActualData,LB_PredictedData);
% 
% CMNN_HeatMap=HeatMap(Confusion_Matrix_NN,'Colormap', 'redgreencmap' ,'RowLabels',[1:52],'ColumnLabels',[1:52]);
%