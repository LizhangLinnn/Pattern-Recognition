clc
clear
close all

%load face data
load face.mat

% config

%% Data Partition

%10-fold crossvalidation
%10 items in each class and 9 data into training set, 1 into test set, same
%as leave-one-out in this case
k=10;                               %Define ratio of partition, k is the proportion sorted into test set
c = cvpartition(l,'Kfold',k);       %Create partition object

%Demonstrate with 1st set
TestIdx=test(c,1);                    %Create index list for test set
TrainingIdx=training(c,1);            %Index list for training set
test=X(:,TestIdx);              
train=X(:,TrainingIdx);

clear c k l X TestIdx TrainingIdx
%% 
test = test';
train = train';



cvalue = 1:2;
tValueStr = '-t 2';  
accuracy = zeros(length(cvalue),1);

%% 

for j = cvalue

    cValueStr = [' -c ',mat2str(j)];
    param = strcat(tValueStr,cValueStr);
    
    decision_val = zeros(52,52);
    for i = 1:52
    % creating different labels for each loop 
        label_test = -ones(52,1);
        label_test(i) = 1;
        label_train = -ones(468,1);
        label_train(((i-1)*9+1):(i*9)) = 1;

    % train and test
        svm_1vAll = fitcsvm(train,label_train,param);
%         SVMModel = fitcsvm(train,Y,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
        [label_result, acc, decision_val(:,i)] = svmpredict(label_test, test, svm_1vAll);
        
        
    end
    [x,result] = max(decision_val,[],1);
    accuracy(j) = 1 - (nnz(result - (1:52))/52);
end




