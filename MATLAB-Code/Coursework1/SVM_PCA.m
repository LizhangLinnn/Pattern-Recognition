clc
clear
close all

%load face data
load face.mat

config

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

%% Projection of raw image data and test data onto eigen space

mean_face = mean(train,2); %return a column vector which is the mean of training data

%compute the covariance matrix
% phi_test = test - mean_face;  %Obtain test data by subtracting from mean face
phi_train = train - mean_face; %Obtain train data
S = (phi_train * phi_train')/size(phi_train,2); %A'A

%compute and normalise the eigenvectors of covariance matrix S
[eig_vec, eig_val] = eig(S);
[~, eig_val_sort_index] = sort(diag(eig_val),'descend');
M_eig_vec = eig_vec(:, eig_val_sort_index(1:467));

train_projection = phi_train' * M_eig_vec;

test_projection = test'* M_eig_vec;



%%
train = train_projection;
test = test_projection;


% cvalue = [1 1000];
% tValueStr = '-t 0';  
% accuracy = zeros(size(cvalue,2),1);
% accuracy = zeros(52,1);
% 
% b = 1;
% for j = cvalue

%     cValueStr = [' -c ',mat2str(j)];
%     param = strcat(tValueStr,cValueStr);
%     
    label_result = zeros(52,52);
    CorrectCount = 0;
    for i = 1:52
    % creating different labels for each loop 
        label_test = -ones(52,1);
        label_test(i) = 1;
        label_train = -ones(468,1);
        label_train(((i-1)*9+1):(i*9)) = 1;

    % train and test
        svm_1vAll = fitcsvm(train,label_train,'Standardize',true,'KernelFunction','linear','KernelScale','auto','BoxConstraint',1);
        [label_result(:,i) , score]= predict(svm_1vAll,test);
        
        [maximum , index] = max(score(:,2)); 
        if(index == i)
            CorrectCount = CorrectCount+1;
        end
        
    end
    accuracy = CorrectCount/size(test,1);
%     b = b+1;
% end







%%
%%
%%
%%

%Examine a scatter plot of the data.
figure
group = cell(27,1);
for i=1:27
    if(i<=9)
    group{i} = 'Class1';
    elseif(i<=19)
    group{i} = 'Class2';
    else
    group{i} = 'Class3';
    end
end
gscatter(train(1:27,1),train(1:27,2),group);
h = gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf face recognition region}');
xlabel('PC1');
ylabel('PC2');
legend('Location','Northwest');

%%
%There are three classes, one of which is linearly separable from the others.

%For each class:

%Create a logical vector (indx) indicating whether an observation is a member of the class.
%Train an SVM classifier using the predictor data and indx.
%Store the classifier in a cell of a cell array.
%It is good practice to define the class order.

SVMModels = cell(3,1);
classes = unique(group);
rng(1); % For reproducibility

for j = 1:numel(classes)
    indx = strcmp(group,classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(train(1:27,1:2),indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end

%%
%SVMModels is a 3-by-1 cell array, with each cell containing a ClassificationSVM classifier. For each cell, the positive class is setosa, versicolor, and virginica, respectively.

%Define a fine grid within the plot, and treat the coordinates as new observations from the distribution of the training data. Estimate the score of the new observations using each classifier.

d = 2;
[x1Grid,x2Grid] = meshgrid(min(train(1:27,1)):d:max(train(1:27,1)),...
    min(train(1:27,2)):d:max(train(1:27,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

%%
for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end
%Each row of Scores contains three scores. The index of the element with the largest score is the index of the class to which the new class observation most likely belongs.

%Associate each new observation with the classifier that gives it the maximum score.
[~,maxScore] = max(Scores,[],2);
%Color in the regions of the plot based on which class the corresponding new observation belongs.

%%
figure
h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.2 1 1; 1 0.2 1; 1 1 0.2]);
hold on
h(4:6) = gscatter(train(1:27,1),train(1:27,2),group);
title('{\bf Iris Classification Regions}');
xlabel('PC2');
ylabel('PC1');
legend(h,{'ClassOne Region','ClassTwo Region','ClassThree Region',...
    'observed ClassOne','observed ClassTwo','observed ClassThree'},...
    'Location','Northwest');
axis tight
hold off


