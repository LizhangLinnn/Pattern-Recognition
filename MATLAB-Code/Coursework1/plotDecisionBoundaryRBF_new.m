%function plotDecisionBoundaryRBF_new(KernelFunction,KernelScale,c,trainData, classA, classB, testData, classC)
%KernelFunction is of char data type. Available values are 'linear','polynomial' and 'rbf'
%traindata has feature vectors as rows
%input classA, classB and classC data types must be char.

KernelFunction = 'rbf';
KernelScale = 50;
c = 100;
trainData = train(1:27,1:4);
testData = test(1:3,1:4);
% trainData = train;
% testData = test;

group = cell(30,1);
for i=1:3
    for j=1:9
        group{(i-1)*9+j} = int2str(i);
    end
end
for i=28:30
   group{i} = int2str(i-27); 
end

% group = cell(520,1);
% for i=1:52
%     for j=1:9
%         group{(i-1)*9+j} = int2str(i);
%     end
% end
% for i=469:520
%    group{i} = int2str(i-468); 
% end

% gscatter(vertcat(trainData(:,1),testData(:,1)),...
%     vertcat(trainData(:,2),testData(:,2)),group);
% h = gca;
% lims = [h.XLim h.YLim]; % Extract the x and y axis limits
% title('{\bf face recognition region}');
% xlabel('PC1');
% ylabel('PC2');
% legend('Location','Northwest');



%%
%1 vs All

    SVMModels = cell(3,1);
    classes = unique(group(1:size(trainData,1)));
    rng(1); % For reproducibility

    for j = 1:numel(classes)
        indx = strcmp(group(1:size(trainData,1)),classes(j)); % Create binary classes for each classifier
        SVMModels{j} = fitcsvm(trainData,indx,'ClassNames',[false true],'Standardize',true,...
            'KernelFunction',KernelFunction,...
            'KernelScale',KernelScale,...
            'BoxConstraint',c);
    end
    
%     SVMModels = cell(52,1);
%     classes = unique(group(1:size(trainData,1)));
%     rng(1); % For reproducibility
% 
%     for j = 1:numel(classes)
%         indx = strcmp(group(1:size(trainData,1)),classes(j)); % Create binary classes for each classifier
%         SVMModels{j} = fitcsvm(trainData,indx,'ClassNames',[false true],'Standardize',true,...
%             'KernelFunction',KernelFunction,...
%             'KernelScale',KernelScale,...
%             'BoxConstraint',c);
%     end
%%
%SVMModels is a 3-by-1 cell array, with each cell containing a ClassificationSVM classifier. For each cell, the positive class is setosa, versicolor, and virginica, respectively.

%Define a fine grid within the plot, and treat the coordinates as new observations from the distribution of the training data. Estimate the score of the new observations using each classifier.

d = 100;
% for i=1:numel(classes)
[x1Grid,x2Grid] = meshgrid(min(vertcat(trainData(:,1),testData(:,1))):d:max(vertcat(trainData(:,1),testData(:,1))),...
    min(vertcat(trainData(:,2),testData(:,2))):d:max(vertcat(trainData(:,2),testData(:,2)))...
    );
xGrid = [x1Grid(:),x2Grid(:)];

%%
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

% for j = 1:numel(classes)
%     [~,score] = predict(SVMModels{j},xGrid);
%     Scores(:,j) = score(:,2); % Second column contains positive-class scores
% end
for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end
%Each row of Scores contains three scores. The index of the element with the largest score is the index of the class to which the new class observation most likely belongs.

%Associate each new observation with the classifier that gives it the maximum score.
[~,maxScore] = max(Scores,[],2);
%Color in the regions of the plot based on which class the corresponding new observation belongs.
done = 1;
%%

gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);
%%
hold on
h(1:3) = gscatter(vertcat(trainData(1:18,1),testData(10,1)),...
    vertcat(trainData(1:18,2),testData(10,2)),...
    vertcat(group(1:18),group(468+10)));

%%
hold on
gscatter(vertcat(trainData(1:18,1),testData(10,1)),...
    vertcat(trainData(1:18,2),testData(10,2)),...
    group,[0 0.5 1; 1 0 0;0 1 0],['.' '.' 's'],[36 36 36]);
% scatter(testData(:,1),testData(:,2),'s',40);
xlabel('PC1');
ylabel('PC2');
% classAregion = strcat(classA,' region');
% classBregion = strcat(classB,' region');
legend(h,{...
    classA,classB,classC});
% axis tight
axis([x1Grid(1),x1Grid(end),x2Grid(1),x2Grid(end)]);
hold off
%end
