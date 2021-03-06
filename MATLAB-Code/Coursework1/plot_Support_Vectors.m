X = train(1:27,1:2);
Y = {'1', '1', '1', '1', '1', '1', '1', '1', '1',...
    '2', '2', '2', '2', '2', '2', '2', '2', '2',...
    '3', '3', '3', '3', '3', '3' ,'3' ,'3' ,'3'};
Y=Y';

figure
gscatter(X(:,1),X(:,2),Y);
h = gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf Scatter Diagram of Training Class 1, 2 & 3}');
xlabel('PC2');
ylabel('PC1');

%%
% Create a logical vector (indx) indicating whether an observation is a member of the class.
% Train an SVM classifier using the predictor data and indx.
% Store the classifier in a cell of a cell array.
% It is good practice to define the class order.

SVMModels = cell(3,1);
classes = unique(Y);
rng(1); % For reproducibility

for j = 1:numel(classes);
    indx = strcmp(Y,classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end

%Define a fine grid within the plot, and treat the coordinates as new observations from the distribution of the training data. Estimate the score of the new observations using each classifier.

d = 5;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes);
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end
%Each row of Scores contains three scores. The index of the element with the largest score is the index of the class to which the new class observation most likely belongs.
%Associate each new observation with the classifier that gives it the maximum score.
[~,maxScore] = max(Scores,[],2);

%Color in the regions of the plot based on which class the corresponding new observation belongs.
figure
h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.25 0 0; 0 0.25 0; 0 0 0.25]);
hold on
h(4:6) = gscatter(X(:,1),X(:,2),Y);
hold on
h(7) = plot(train(SVMModels{1}.IsSupportVector,1),...
    train(SVMModels{1}.IsSupportVector,2),'ko','LineWidth', 2, 'MarkerSize',20);
h(8) = plot(train(SVMModels{2}.IsSupportVector,1),...
    train(SVMModels{2}.IsSupportVector,2),'s','LineWidth', 2, 'MarkerSize',13);
h(9) = plot(train(SVMModels{3}.IsSupportVector,1),...
    train(SVMModels{3}.IsSupportVector,2),'>','LineWidth', 2, 'MarkerSize',15);

title('{\bf Training Class 1, 2 & 3 RBF result}');
xlabel('PC1');
ylabel('PC2');
legend(h,{'Class One Region','Class Two Region','Class Three Region',...
    'Observed Class One','Observed Class Two','Observed Class Three',...
    'SVs - Class One','SVs - Class Two','SVs - Class Three'});
axis tight
hold off
