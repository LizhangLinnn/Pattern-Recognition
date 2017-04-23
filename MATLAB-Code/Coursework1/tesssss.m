trainData = train(1:18,1:2);
testData = test(2,1:2);
classA = 'class 1';
classB = 'class 2';
classC = 'predicted class';
KernelFunction = 'linear';
%KernelFunction is of char data type. Available values are 'linear','polynomial' and 'rbf'
%traindata has feature vectors as rows
%input classA and classB data types must be char.

figure
group = cell(19,1);
for i=1:19
    if(i<=9)
    group{i} = classA;
    elseif(i<=18)
    group{i} = classB;
    else
    group{i} = classC;
    end
end
gscatter(vertcat(trainData(:,1),testData(:,1)),...
    vertcat(trainData(:,2),testData(:,2)),group);
h = gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf face recognition region}');
xlabel('PC1');
ylabel('PC2');
% legend('Location','Northwest');

%%
%There are three classes, one of which is linearly separable from the others.

%For each class:

%Create a logical vector (indx) indicating whether an observation is a member of the class.
%Train an SVM classifier using the predictor data and indx.
%Store the classifier in a cell of a cell array.
%It is good practice to define the class order.

    SVMModels = cell(2,1);
    classes = unique(group);
    rng(1); % For reproducibility

    for j = 1:numel(classes)
        indx = strcmp(group,classes(j)); % Create binary classes for each classifier
        SVMModels{j} = fitcsvm(trainData,indx,'ClassNames',[false true],'Standardize',true,...
            'KernelFunction',KernelFunction,'BoxConstraint',1);
    end
%%
%SVMModels is a 3-by-1 cell array, with each cell containing a ClassificationSVM classifier. For each cell, the positive class is setosa, versicolor, and virginica, respectively.

%Define a fine grid within the plot, and treat the coordinates as new observations from the distribution of the training data. Estimate the score of the new observations using each classifier.

d = 2;
[x1Grid,x2Grid] = meshgrid(min(trainData(:,1)):d:max(trainData(:,1)),...
    min(trainData(:,2)):d:max(trainData(:,2)));
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
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.25 0 0; 0 0 0.25]);
%%
hold on
h(3:4) = gscatter(trainData(:,1),trainData(:,2),group,[1 0 0; 0 0 1],'.',36);
% hold on
% h(5:6) = gscatter(testData(:,1),testData(:,2),{'TestData'},'.',48);
xlabel('PC1');
ylabel('PC2');
classAregion = strcat(classA,' region');
classBregion = strcat(classB,' region');
legend(h,{classAregion,classBregion,...
    classA,classB});
axis tight
hold off