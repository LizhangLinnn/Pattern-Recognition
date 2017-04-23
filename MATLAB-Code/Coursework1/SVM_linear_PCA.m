clc
clear
close all

%load face data
% load face.mat

load data.mat
config

%% 


mean_face = mean(train,2); %return a column vector which is the mean of training data
mean_face = repmat(mean_face,[1,468]);
%compute the covariance matrix
% phi_test = test - mean_face;  %Obtain test data by subtracting from mean face
phi_train = train - mean_face; %Obtain train data
S = (phi_train * phi_train')/size(phi_train,2); %A'A

%compute and normalise the eigenvectors of covariance matrix S
[eig_vec, eig_val] = eig(S);
[~, eig_val_sort_index] = sort(diag(eig_val),'descend');
M_eig_vec = eig_vec(:, eig_val_sort_index(1:60));

train_projection = phi_train' * M_eig_vec;

test_projection = test'* M_eig_vec;

train = train_projection;
test = test_projection;

%% 



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
% 5.5
tic

% accuracy = zeros(size(crange,2),1);
% a=1;
% for c = crange
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
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint',1, ...
    'Standardize', true, ...
    'ClassNames', [1; -1]);

    [label,score] = predict(svm_1vAll,test);
    decision_val(:,i) = score(:,1);
    end


    [x,result] = max(decision_val);
    accuracy = 1 - (nnz(result - (1:52))/52);
%     a = a+1;

% end

toc

%%
numPtsInGrid = 3;

LBs = zeros(size(train,2),1);
UBs = zeros(size(train,2),1);
Grid = 0;
for i=1:size(train,2)-1
   LBs(i) = min(vertcat(train(:,i),test(:,i)));
   UBs(i) = max(vertcat(train(:,i),test(:,i)));
   Range1 = linspace(LBs(i), UBs(i), numPtsInGrid);
   
   LBs(i+1) = min(vertcat(train(:,i+1),test(:,i+1)));
   UBs(i+1) = max(vertcat(train(:,i+1),test(:,i+1)));
   Range2 = linspace(LBs(i+1), UBs(i+1), numPtsInGrid);
   
   [X1, X2] = meshgrid(Range1, Range2);
%1
   A1 = X1';
   A1 = A1(:);
   A2 = X2';
   A2 = A2(:);
   
   if(Grid == 0)
       Grid = [X1(:),X2(:)];
   elseif(A1 == Grid(:,end))
      Grid = horzcat(Grid, [A2(:)]);
%    elseif(mode(i/2)==0)
%        X2 = X2';
%        Grid = horzcat(Grid, [X2(:)]);
   elseif(X1(:) == Grid(:,end))
       Grid = horzcat(Grid, [X2(:)]);
   end
   
end


%%
tic


N = size(Grid,1);
Scores = zeros(N,52);

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
'KernelScale', 50, ...
'BoxConstraint',10, ...
'Standardize', true, ...
'ClassNames', [1; -1]);

[~,score] = predict(svm_1vAll,Grid);
Scores(:,i)  = score(:,1);
end
toc
[~,maxScore] = max(Scores,[],2);
%%

gscatter(Grid(:,1),Grid(:,2),maxScore...
    );


%% original graph pc1 vs pc2
figure;
group = 0;
for i=1:52
    if (group==0)
        group = [i i i i i i i i i]';
    else
        group=vertcat(group,[i i i i i i i i i]');
    end
end
gscatter(train(:,1),train(:,2),group);

%%
contourf(X1(:), X2(:), Z(:))


%%

%% linear 1vsALL success & failure
%failure case
figure;
subplot(121);
plotDecisionBoundaryRBF('rbf',60,50,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('Failure Case - Test Sample 7');
%success case
subplot(122);
plotDecisionBoundary('rbf',60,50,vertcat(train(10:18,1:2),train(1:9,1:2)),'class 2','class 1',test(2,1:2),'Test Sample 2');
title('Success Case - Test Sample 2');

%% Poly 1vsALL c & Degree comparison
figure;
subplot(321);
plotDecisionBoundaryPoly('polynomial',3,0.0001,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('degree = 3, C = 0.0001');
subplot(322);
plotDecisionBoundaryPoly('polynomial',6,0.0001,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('degree = 6, C = 0.0001');
subplot(323);
plotDecisionBoundaryPoly('polynomial',3,1,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('degree = 3, C = 1');
subplot(324);
plotDecisionBoundaryPoly('polynomial',6,1,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('degree = 6, C = 1');
subplot(325);
plotDecisionBoundaryPoly('polynomial',3,10,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('degree = 3, C = 10');
subplot(326);
plotDecisionBoundaryPoly('polynomial',6,10,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('degree = 6, C = 10');

%% RBF 1vsALL c & g comparison
figure;
subplot(321);
plotDecisionBoundaryRBF('rbf',12,1,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('KernelScale = 12, C = 1, accuracy = 0.5577');
subplot(322);
plotDecisionBoundaryRBF('rbf',12,100,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('KernelScale = 12, C = 100, accuracy = 0.5577');
subplot(323);
plotDecisionBoundaryRBF('rbf',39,1,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('KernelScale = 39, C = 1, accuracy = 0.9231');
subplot(324);
plotDecisionBoundaryRBF('rbf',39,100,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('KernelScale = 39, C = 100, accuracy = 0.9231');
subplot(325);
plotDecisionBoundaryRBF('rbf',72,1,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('KernelScale = 72, C = 1, accuracy = 0.8846');
subplot(326);
plotDecisionBoundaryRBF('rbf',72,100,vertcat(train(6*9+1:7*9,1:2),train(33*9+1:34*9,1:2)),'class 7','class 34',test(7,1:2), 'Test Sample 7');
title('KernelScale = 72, C = 100, accuracy = 0.8846');

