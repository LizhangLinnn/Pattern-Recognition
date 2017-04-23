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

clear c k l X TestIdx TrainingIdx

test = test';
train = train';

%%

label_result = zeros(52,52);
decision_val = zeros(52,52);
b = 1;
% compensation = 0.001:0.3:6;
compensation = 0.001:1:5;
accuracy = zeros(size(compensation,2),1);
for c = compensation
    CorrectCount=0;
    for i = 1:52
    % creating different labels for each loop 
        label_test = -ones(52,1);
        label_test(i) = 1;
        label_train = -ones(468,1);
        label_train(((i-1)*9+1):(i*9)) = 1;

    % train and test
        svm_1vAll = fitcsvm(train,label_train,'Standardize',true,'KernelFunction','linear','KernelScale','auto','BoxConstraint',c);
        [label_result(:,i) , score]= predict(svm_1vAll,test);
        %accuracy
        [minimum , index] = min(score(:,1)); 
        if(index == i)
            CorrectCount = CorrectCount+1;
        end
        
    end
    accuracy(b) = CorrectCount/size(test,1);
    b=b+1;
end
%     [x,result] = max(decision_val);
%     accuracy = 1 - (nnz(result - (1:52))/52);
%     b = b+1;
% end
% end
%%
% sv = svm_1vAll.SupportVectors;
% figure
% gscatter(train(:,1),train(:,2),label_train)
% hold on
% plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
% legend('1','-1','Support Vector')
% hold off


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
title('{\bf Scatter Diagram of Iris Measurements}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
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

d = 0.2;
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






%% testing for plot
load fisheriris
X = meas(:,3:4);
Y = species;

%%
figure
gscatter(X(:,1),X(:,2),Y);
h = gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf Scatter Diagram of Iris Measurements}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend('Location','Northwest');
%%
SVMModels = cell(3,1);
classes = unique(Y);
rng(1); % For reproducibility

for j = 1:numel(classes);
    indx = strcmp(Y,classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end
%%
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes);
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end
[~,maxScore] = max(Scores,[],2);

%%
figure
h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.3 1 1; 1 0.3 1; 1 1 0.3]);
hold on
h(4:6) = gscatter(X(:,1),X(:,2),Y);
title('{\bf Iris Classification Regions}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend(h,{'setosa region','versicolor region','virginica region',...
    'observed setosa','observed versicolor','observed virginica'},...
    'Location','Northwest');
axis tight
hold off