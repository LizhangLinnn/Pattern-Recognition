load wine.data.csv

%% data partition
% Split the data into 3 sets for Training (118), Validation (20), 
% and Testing (42). Move the class identifiers to separate vectors
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



%% A
load train_validate.mat
% A.b
% Normalizing feature vectors to unit norm L2 
load test.mat
data = vertcat(train_validate,test);
train_validate_norm = normc(data(:,2:14));
train_validate_norm = horzcat(train_validate(:,1),train_validate_norm(1:138,:));
test_norm = normc(data(:,2:14));
test_norm = horzcat(test(:,1),test_norm(139:178,:));

%% B
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

%% C
%% find minimum covariance group
cov_average = zeros(13,13);
for i=1:13
    
    for j=1:13
        if(j>=i)
        cov_average(i,j) = (cov_ClassOne_norm(i,j)*cov_ClassTwo_norm(i,j)*cov_ClassThree_norm(i,j))^(1/3);
        else
            cov_average(i,j) = 100;
        end
    end
end
[mini,index ]= min(abs(cov_average));
%minimum.. cov: row 8, column 11
%% mean & cov - 1:13
subplot(121);
group_bar = zeros(13,3);
for i=1:13
    group_bar(i,1) = cov_ClassOne(i,i);
    group_bar(i,2) = cov_ClassTwo(i,i);
    group_bar(i,3) = cov_ClassThree(i,i);
end
bar([1:13],group_bar(:,:));
title('Variances of Different Dimensions within All Classes');
xlabel('dimension');
set(gca,'YScale','log')
ylabel('variance');
legend('Class 1','Class 2','Class 3');
%mean
subplot(122);
group_bar = zeros(13,3);
for i=1:13
    group_bar(i,1) = mean_ClassOne(i);
    group_bar(i,2) = mean_ClassTwo(i);
    group_bar(i,3) = mean_ClassThree(i);
end
ax = bar([1:13],group_bar(:,:));
title('Mean of Different Dimensions within All Classes');
xlabel('dimension');
set(gca,'YScale','log')
ylabel('mean');
legend('Class 1','Class 2','Class 3');

%% mean & cov - 1:13 normalised
subplot(121);
group_bar = zeros(13,3);
for i=1:13
    group_bar(i,1) = cov_ClassOne_norm(i,i);
    group_bar(i,2) = cov_ClassTwo_norm(i,i);
    group_bar(i,3) = cov_ClassThree_norm(i,i);
end
bar([1:13],group_bar(:,:));
title('Variances of Different Dimensions within All Classes');
xlabel('dimension');
% set(gca,'YScale','log')
ylabel('variance');
legend('Class 1','Class 2','Class 3');
%mean
subplot(122);
group_bar = zeros(13,3);
for i=1:13
    group_bar(i,1) = mean_ClassOne_norm(i);
    group_bar(i,2) = mean_ClassTwo_norm(i);
    group_bar(i,3) = mean_ClassThree_norm(i);
end
ax = bar([1:13],group_bar(:,:));
title('Mean of Different Dimensions within All Classes');
xlabel('dimension');
% set(gca,'YScale','log')
ylabel('mean');
legend('Class 1','Class 2','Class 3');


%% cov unnormalised
% cov_ClassOne_norm_plot = cov_ClassOne_norm;
% cov_ClassTwo_norm_plot = cov_ClassTwo_norm;
% cov_ClassThree_norm_plot = cov_ClassThree_norm;
% for i =1:13
%    cov_ClassTwo_norm_plot(i,i) = 0;
%       cov_ClassOne_norm_plot(i,i) = 0;
%          cov_ClassThree_norm_plot(i,i) = 0;
% end
subplot(131);surf(cov_ClassOne);
title('Covariance Within Class One');
xlabel('Dimension');
ylabel('Dimension');
subplot(132);surf(cov_ClassTwo);
title('Covariance Within Class Two');
xlabel('Dimension');
ylabel('Dimension');
subplot(133);surf(cov_ClassThree);
title('Covariance Within Class Three');
xlabel('Dimension');
ylabel('Dimension');

%% cov norm
% cov_ClassOne_norm_plot = cov_ClassOne_norm;
% cov_ClassTwo_norm_plot = cov_ClassTwo_norm;
% cov_ClassThree_norm_plot = cov_ClassThree_norm;
% for i =1:13
%    cov_ClassTwo_norm_plot(i,i) = 0;
%       cov_ClassOne_norm_plot(i,i) = 0;
%          cov_ClassThree_norm_plot(i,i) = 0;
% end
subplot(141);surf(cov_ClassOne_norm);
title('Covariance Within Class One');
xlabel('Dimension');
ylabel('Dimension');
zlim([-0.5*10^-3,10^(-3)]);
subplot(142);surf(cov_ClassTwo_norm);
title('Covariance Within Class Two');
xlabel('Dimension');
ylabel('Dimension');
zlim([-0.5*10^-3,10^(-3)]);
subplot(143);surf(cov_ClassThree_norm);
title('Covariance Within Class Three');
xlabel('Dimension');
ylabel('Dimension');
zlim([-0.5*10^-3,10^(-3)]);
subplot(144);surf(cov_All_norm);
title('Covariance Within Class Three');
xlabel('Dimension');
ylabel('Dimension');
% zlim([-0.5*10^-3,10^(-3)]);
%% means
% mean_ClassOne_norm_plot = mean_ClassOne_norm;
% mean_ClassTwo_norm_plot = mean_ClassTwo_norm;
% cov_ClassThree_norm_plot = cov_ClassThree_norm;
% for i =1:13
%    cov_ClassTwo_norm_plot(i,i) = 0;
%       cov_ClassOne_norm_plot(i,i) = 0;
%          cov_ClassThree_norm_plot(i,i) = 0;
% end
mean_distance = zeros(13,13);
for i=1:13
    for j=1:13
        mean_distance(i,j) = (mean_ClassOne_norm(i)-mean_ClassTwo_norm(i))^2+...
            (mean_ClassThree_norm(i)-mean_ClassTwo_norm(i))^2+...
            (mean_ClassOne_norm(i)-mean_ClassThree_norm(i))^2+...
            (mean_ClassOne_norm(j)-mean_ClassTwo_norm(j))^2+...
            (mean_ClassThree_norm(j)-mean_ClassTwo_norm(j))^2+...
            (mean_ClassOne_norm(j)-mean_ClassThree_norm(j))^2;
    end
end
%% coefficient of variance 

%%
figure;surf(mean_distance);
title('squared distance between mean points of three classes');
xlabel('Dimension');
ylabel('Dimension');

%% raw data
% %class one dim 1 & 2
figure;
dim1 = 5;
dim2 = 13;


dim1 = dim1 +1;
dim2 = dim2+1;

% ***************** incorrect test data *****************
% data = vertcat(train_validate,incorrect_L2);
% gscatter(data(:,dim1),data(:,dim2),data(:,1));

% ***************** training_ validation *****************
h1(1:3) = gscatter(train_validate(1:118,dim1),train_validate(1:118,dim2),train_validate(1:118,1));
hold on

% scatter(mean_ClassOne(dim1-1),mean_ClassOne(dim2-1),'s');
% scatter(mean_ClassTwo(dim1-1),mean_ClassTwo(dim2-1),'s');
% scatter(mean_ClassThree(dim1-1),mean_ClassThree(dim2-1),'s');

plot_cov_ClassOne = [cov_ClassOne(dim1-1,dim1-1),cov_ClassOne(dim1-1,dim2-1);...
    cov_ClassOne(dim2-1,dim1-1), cov_ClassOne(dim2-1,dim2-1)];
plot_cov_ClassTwo = [cov_ClassTwo(dim1-1,dim1-1),cov_ClassTwo(dim1-1,dim2-1);...
    cov_ClassTwo(dim2-1,dim1-1), cov_ClassTwo(dim2-1,dim2-1)];
plot_cov_ClassThree = [cov_ClassThree(dim1-1,dim1-1),cov_ClassThree(dim1-1,dim2-1);...
    cov_ClassThree(dim2-1,dim1-1), cov_ClassThree(dim2-1,dim2-1)];
 
h = error_ellipse(plot_cov_ClassOne,[mean_ClassOne(dim1-1),mean_ClassOne(dim2-1)],'conf',0.80);
h = error_ellipse(plot_cov_ClassTwo,[mean_ClassTwo(dim1-1),mean_ClassTwo(dim2-1)],'conf',0.80);
h = error_ellipse(plot_cov_ClassThree,[mean_ClassThree(dim1-1),mean_ClassThree(dim2-1)],'conf',0.80);

% 
% r=0.02;
% ang=0:0.01:2*pi; 
% xp=r*cos(ang);
% yp=r*sin(ang);
% plot(mean_ClassOne_norm(dim1-1)+xp,mean_ClassOne_norm(dim2-1)+yp);
% 
% 
% ang=0:0.01:2*pi; 
% xp=r*cos(ang);
% yp=r*sin(ang);
% plot(mean_ClassTwo_norm(dim1-1)+xp,mean_ClassTwo_norm(dim2-1)+yp);
% 
% 
% ang=0:0.01:2*pi; 
% xp=r*cos(ang);
% yp=r*sin(ang);
% plot(mean_ClassThree_norm(dim1-1)+xp,mean_ClassThree_norm(dim2-1)+yp);

%labels and titles
% axis equal;
axis tight;

% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% %L2
% %misclassified to class 3
% h1(4:6) = gscatter(incorrect_L2(:,dim1),incorrect_L2(:,dim2),incorrect_L2(:,1));% failure case no. 5
% %correctly predicted
% h1(7:9) = gscatter(correct_L2(:,dim1),correct_L2(:,dim2),correct_L2(:,1)); %success case no. 1
% % zoom in
% limits = [1.88, 3.26,390, 873];
% axis(limits)

%L1
%misclassified to class 3
h1(4:6) = gscatter(incorrect_L1(:,dim1),incorrect_L1(:,dim2),incorrect_L1(:,1));% failure case no. 5
%correctly predicted
h1(7:9) = gscatter(correct_L1(:,dim1),correct_L1(:,dim2),correct_L1(:,1)); %success case no. 1
%plot data that was not correctly predicted by L2 but correct predicted by L1
j=0;k=0;
L1_success_L2_fail = [];
for i=1:size(incorrect_L2)
    j=j+1;
    if(isequal(incorrect_L1(j,:),incorrect_L2(i,:))==0)
        j= j-1;
        k = k+1;
        L1_success_L2_fail(k,:) = incorrect_L2(i,:);
    end
end
h1(10) = scatter(L1_success_L2_fail(:,dim1),L1_success_L2_fail(:,dim2),L1_success_L2_fail(:,1)); %only class 2 & 3
set(h1(10),'marker','o');
set(h1(10),'sizedata',200);
set(h1(10),'linewidth',3);
% zoom in
limits = [1.88, 3.26, 390, 873];
axis(limits)


set(h1(1:3),'markersize',8);
set(h1(4:6),'marker','x');
set(h1(4:6),'markersize',10);
set(h1(4:6),'linewidth',2);
set(h1(7:9),'marker','+');
set(h1(7:9),'markersize',10);
set(h1(7:9),'linewidth',2);
legend(h1,{'Class 1','Class 2','Class 3',...
    'Class 1 failure','Class 2 failure','Class 3 failure'...
    ,'Class 1 successful','Class 2 successful','Class 3 successful','Corrected by L1'});
  
xlabel(strcat('dimension ',int2str(dim1-1)));
ylabel(strcat('dimension ',int2str(dim2-1)));
title(strcat('L1 - Successful and Failure Test Data Points in Dimension ',int2str(dim1-1),' and ',int2str(dim2-1)));
hold off;



%% normalised
% %class one dim 1 & 2
figure;
dim1 = 7;
dim2 = 6;


dim1 = dim1 +1;
dim2 = dim2+1;

% ***************** incorrect test data *****************
% data = vertcat(train_validate,incorrect_L2);
% gscatter(data(:,dim1),data(:,dim2),data(:,1));

% ***************** training_ validation *****************
% h1(1:3) = gscatter(train_validate(:,dim1),train_validate(:,dim2),train_validate(:,1));
h1(1:3) = gscatter(train_validate_norm(:,dim1),train_validate_norm(:,dim2),train_validate(:,1));
hold on

% scatter(mean_ClassOne_norm(dim1-1),mean_ClassOne_norm(dim2-1),'*');
% scatter(mean_ClassTwo_norm(dim1-1),mean_ClassTwo_norm(dim2-1),'*');
% scatter(mean_ClassThree_norm(dim1-1),mean_ClassThree_norm(dim2-1),'*');


plot_cov_ClassOne_norm = [cov_ClassOne_norm(dim1-1,dim1-1),cov_ClassOne_norm(dim1-1,dim2-1);...
    cov_ClassOne_norm(dim2-1,dim1-1), cov_ClassOne_norm(dim2-1,dim2-1)];
plot_cov_ClassTwo_norm = [cov_ClassTwo_norm(dim1-1,dim1-1),cov_ClassTwo_norm(dim1-1,dim2-1);...
    cov_ClassTwo_norm(dim2-1,dim1-1), cov_ClassTwo_norm(dim2-1,dim2-1)];
plot_cov_ClassThree_norm = [cov_ClassThree_norm(dim1-1,dim1-1),cov_ClassThree_norm(dim1-1,dim2-1);...
    cov_ClassThree_norm(dim2-1,dim1-1), cov_ClassThree_norm(dim2-1,dim2-1)];

plot_cov_All_norm = [cov_All_norm(dim1-1,dim1-1),cov_All_norm(dim1-1,dim2-1);...
    cov_All_norm(dim2-1,dim1-1), cov_All_norm(dim2-1,dim2-1)];

h = error_ellipse(plot_cov_ClassOne_norm,[mean_ClassOne_norm(dim1-1),mean_ClassOne_norm(dim2-1)],'conf',0.2);
% h = error_ellipse(plot_cov_ClassTwo_norm,[mean_ClassTwo_norm(dim1-1),mean_ClassTwo_norm(dim2-1)],'conf',0.3);
set(h,'linewidth',2);

% h = error_ellipse(plot_cov_ClassThree_norm,[mean_ClassThree_norm(dim1-1),mean_ClassThree_norm(dim2-1)],'conf',0.7);
% 
% h = error_ellipse(plot_cov_All_norm,[mean_All_norm(dim1-1),mean_All_norm(dim2-1)],'conf',0.7);
% 
% euclidean
r=0.01;
ang=0:0.01:2*pi; 
xp=r*cos(ang);
yp=r*sin(ang);
h = plot(mean_ClassOne_norm(dim1-1)+xp,mean_ClassOne_norm(dim2-1)+yp);
set(h,'linewidth',2);



% 
% 
% ang=0:0.01:2*pi; 
% xp=r*cos(ang);
% yp=r*sin(ang);
% plot(mean_ClassTwo_norm(dim1-1)+xp,mean_ClassTwo_norm(dim2-1)+yp);
% 
% 
% ang=0:0.01:2*pi; 
% xp=r*cos(ang);
% yp=r*sin(ang);
% plot(mean_ClassThree_norm(dim1-1)+xp,mean_ClassThree_norm(dim2-1)+yp);

%labels and titles
axis equal;
axis tight;

% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% %L1
% % Three class two test data misclassified
% h1(4) = scatter(incorrect_L1_norm(:,dim1),incorrect_L1_norm(:,dim2),incorrect_L1_norm(:,1),'g');% failure case no. 5
% 
% % NN training data
% h1(5) = scatter(train_validate_norm([35,94,27], dim1),train_validate_norm([35,94,27], dim2));

% input = 35;input2 =1;
%     d = (incorrect_L1_norm(input2,dim1)-train_validate_norm(input,dim1))/100;
%     x1= train_validate_norm(input,dim1):d:incorrect_L1_norm(input2,dim1);
%     d = (incorrect_L1_norm(input2,dim2)-train_validate_norm(input,dim2))/100;
%     x2= train_validate_norm(input,dim2):d:incorrect_L1_norm(input2,dim2);
% h1(6) = scatter(x1,x2,4);
% 
% input = 94;input2 =2;
%     d = (incorrect_L1_norm(input2,dim1)-train_validate_norm(input,dim1))/100;
%     x1= train_validate_norm(input,dim1):d:incorrect_L1_norm(input2,dim1);
%     d = (incorrect_L1_norm(input2,dim2)-train_validate_norm(input,dim2))/100;
%     x2= train_validate_norm(input,dim2):d:incorrect_L1_norm(input2,dim2);
% h1(7) = scatter(x1,x2,4);
% 
% input = 27;input2 =3;
%     d = (incorrect_L1_norm(input2,dim1)-train_validate_norm(input,dim1))/100;
%     x1= train_validate_norm(input,dim1):d:incorrect_L1_norm(input2,dim1);
%     d = (incorrect_L1_norm(input2,dim2)-train_validate_norm(input,dim2))/100;
%     x2= train_validate_norm(input,dim2):d:incorrect_L1_norm(input2,dim2);
% h1(8) = scatter(x1,x2,4);
set(h1(1:3),'markersize',8);
% set(h1(4),'marker','x');
% set(h1(4),'SizeData',300);
% set(h1(4),'linewidth',2);
% 
% set(h1(5),'marker','s');
% set(h1(5),'SizeData',300);
% set(h1(5),'linewidth',2);
%
% legend(h1(1:5),{'Class 1','Class 2','Class 3',...
%     'failure cases','assigned training data'...
%     });
%   'Class 1 successful','Class 2 successful','Class 3 successful'});
xlabel(strcat('dimension ',int2str(dim1-1)));
ylabel(strcat('dimension ',int2str(dim2-1)));
% title(strcat('L1 (Normalised) - Failure Test Data points in Dimension ',int2str(dim1-1),' and ',int2str(dim2-1)));
legend('Class 1','Class 2','Class 3','Mahalanobis Distance with Cov of Class 1',...
    'euclidean distance');
hold off;

%% KNN ********************************************
%% Initialisation

% 
% load class_label.mat
% load wine.mat
% load all.mat
load train_validate.mat
load test.mat
% %% Normalise training data
train_validate_norm = train_validate;
train_validate_norm(:,2:14) = normc(train_validate(:,2:14));
test_norm = normc(test(:,2:14));
test_norm = horzcat(test(:,1),test_norm);
%% L1 - original

% Train model
[L1, validate_L1] = trainClassifier_L1(train_validate(1:118,:));
yfit_L1 = L1.predictFcn(test(:,2:14));
result = yfit_L1 - test(:,1);
j=0; k=0;
incorrect_L1 = [];
correct_L1 = [];
for i=1:size(test,1)
     if(result(i) ~= 0)
         j=j+1;
         incorrect_L1(j,:) = test(i,:);
     else
         k=k+1;
         correct_L1(k,:) = test(i,:);
     end
end

% Test with testing data
accuracy_L1 = 1-(nnz(yfit_L1 - test(:,1))/size(yfit_L1 - test(:,1),1));
%% L1 - normalised

% Train model
[L1_norm, validate_L1_norm] = trainClassifier_L1(train_validate_norm(1:118,:));
yfit_L1_norm = L1_norm.predictFcn(test_norm(:,2:14));
result = yfit_L1_norm - test_norm(:,1);
j=0; k=0;
incorrect_L1_norm = [];
correct_L1_norm = [];
for i=1:size(test_norm,1)
     if(result(i) ~= 0)
         j=j+1;
         incorrect_L1_norm(j,:) = test_norm(i,:);
     else
         k=k+1;
         correct_L1_norm(k,:) = test_norm(i,:);
     end
end
% Test with testing data
accuracy_L1_norm = 1-(nnz(yfit_L1_norm - test_norm(:,1))/size(yfit_L1_norm - test_norm(:,1),1));

%% L2

% Train model
[L2, validate_L2] = trainClassifier_L2_new(train_validate(1:118,:));

yfit_L2 = L2.predictFcn(test(:,2:14));
result = yfit_L2 - test(:,1);
j=0; k=0;
incorrect_L2 = [];
correct_L2 = [];
for i=1:size(test,1)
     if(result(i) ~= 0)
         j=j+1;
         incorrect_L2(j,:) = test(i,:);
     else
         k=k+1;
         correct_L2(k,:) = test(i,:);
     end
end

accuracy_L2= 1-(nnz(yfit_L2 - test(:,1))/size(yfit_L2 - test(:,1),1));
%% L2 - normalised

% Train model
[L2_norm, validate_L2_norm] = trainClassifier_L2_new(train_validate_norm(1:118,:));

yfit_L2_norm = L2_norm.predictFcn(test_norm(:,2:14));
result = yfit_L2_norm - test_norm(:,1);
j=0; k=0;
incorrect_L2_norm = [];
correct_L2_norm = [];
for i=1:size(test_norm,1)
     if(result(i) ~= 0)
         j=j+1;
         incorrect_L2_norm(j,:) = test_norm(i,:);
     else
         k=k+1;
         correct_L2_norm(k,:) = test_norm(i,:);
     end
end
accuracy_L2_norm= 1-(nnz(yfit_L2_norm - test_norm(:,1))/size(yfit_L2_norm - test_norm(:,1),1));

%% Chi2
[accuracy_Chi2, ~] = Chi2(train_validate(1:118,:),test);
%% Chi2 - normalised
[accuracy_Chi2_norm, ~] = Chi2(train_validate_norm(1:118,:),test_norm);

%% Histogram
[accuracy_hist,~] = histogram_intersection_norm(train_validate(1:118,:),test);

%% Histogram normalised
% [accuracy_hist_norm] = histogram_intersection(train_validate_norm(1:118,:),test_norm);
[accuracy_hist_norm,~] = histogram_intersection_norm(train_validate_norm(1:118,:),test_norm);

%% Correlation

% Train model
[Corr, validate_Corr] = trainClassifier_Corr(train_validate(1:118,:));

yfit_Corr = Corr.predictFcn(test(:,2:14));
accuracy_Corr = 1-(nnz(yfit_Corr - test(:,1))/size(yfit_Corr - test(:,1),1));
%% Correlation knn
[n,d]=knnsearch(train_validate(1:118,2:14),test(:,2:14),'k',1,'distance','correlation');
[ncb,dcb] = knnsearch(train_validate(1:118,2:14),test(:,2:14),'k',1,...
   'distance','cityblock');
gscatter(train_validate(1:118,dim1),train_validate(1:118,dim2),train_validate(1:118,1));
line(test(:,dim1),test(:,dim2),'marker','x','color','k',...
   'markersize',10,'linewidth',2,'linestyle','none')
line(train_validate(n,dim1),train_validate(n,dim2),'color',[.5 .5 .5],'marker','o',...
   'linestyle','none','markersize',10)
line(train_validate(ncb,dim1),train_validate(ncb,dim2),'color',[.5 .5 .5],'marker','p',...
   'linestyle','none','markersize',10)
legend('class one','class two','class three','query point','L2','L1')
%accuracy
count =0;
for i=1:40
    label_train = train_validate(n(i),1);
    if (label_train == test(i,1))
       count = count+1; 
    end
end
accuracy_Corr = count/40;

%% Correlation - normalised

% Train model
[Corr_norm, validate_Corr_norm] = trainClassifier_Corr(train_validate_norm(1:118,:));

yfit_Corr_norm = Corr_norm.predictFcn(test_norm(:,2:14));
accuracy_Corr_norm = 1-(nnz(yfit_Corr_norm - test(:,1))/size(yfit_Corr_norm - test(:,1),1));
%% Mahalanobis

% Train model
[Maha, validate_Maha] = trainClassifier_Maha(train_validate(1:118,:),cov_All);

yfit_Maha = Maha.predictFcn(test(:,2:14));
accuracy_Maha = 1-(nnz(yfit_Maha - test(:,1))/size(yfit_Maha - test(:,1),1));
%% Mahalanobis Class 1

% Train model
[Maha, validate_Maha] = trainClassifier_Maha(train_validate(1:118,:),cov_ClassOne);

yfit_Maha = Maha.predictFcn(test(:,2:14));
accuracy_Maha1 = 1-(nnz(yfit_Maha - test(:,1))/size(yfit_Maha - test(:,1),1));

%% Mahalanobis Class 1

[eig_vec, eig_val] = eig(inv(cov_ClassOne));
G = (((eig_val).^0.5)' * eig_vec);
train_validate_maha = train_validate(1:118,2:14)*G;
% predict = knnsearch(train_validate_maha,test(:,2:14)*G,'k',1,'distance','euclidean');

predict = zeros(40,1);
for i=1:40
    d_maha = (G*(test(i,2:14)-train_validate(1:118,2:14))')'* (G*(test(i,2:14)-train_validate(1:118,2:14))');
    d_maha = diag(d_maha);
    [~,predict(i)] = min(d_maha);
%     if(predict(i) <= 39)
%         predict(i) = 1;
%     elseif(predict(i) <= 86)
%         predict(i) = 2;
%     else
%         predict(i) = 3;
%     end
end

CorrectCount = 0;
for i =1:13
   if(predict(i) <= 39)
       CorrectCount = CorrectCount+1;
       
   end
end
for i =14:29
   if(40<=predict(i) <= 86)
       CorrectCount = CorrectCount+1;
   end
end
for i =30:40
   if(87<=predict(i) <= 118)
       CorrectCount = CorrectCount+1;
   end
end
accuracy_Maha1 = CorrectCount/40;
    
%% Maha ClassOne norm
[eig_vec, eig_val] = eig(inv(cov_ClassOne_norm));
G = (((eig_val).^0.5)' * eig_vec);
train_validate_maha = train_validate_norm(1:118,2:14)*G;
% predict = knnsearch(train_validate_maha,test(:,2:14)*G,'k',1,'distance','euclidean');

predict = zeros(40,1);
for i=1:40
    d_maha = (G*(test_norm(i,2:14)-train_validate_norm(1:118,2:14))')'* (G*(test_norm(i,2:14)-train_validate_norm(1:118,2:14))');
    d_maha = diag(d_maha);
    [~,predict(i)] = min(d_maha);
%     if(predict(i) <= 39)
%         predict(i) = 1;
%     elseif(predict(i) <= 86)
%         predict(i) = 2;
%     else
%         predict(i) = 3;
%     end
end

CorrectCount = 0;
for i =1:13
   if(predict(i) <= 39)
       CorrectCount = CorrectCount+1;
       
   end
end
for i =14:29
   if(40<=predict(i) <= 86)
       CorrectCount = CorrectCount+1;
   end
end
for i =30:40
   if(87<=predict(i) <= 118)
       CorrectCount = CorrectCount+1;
   end
end
accuracy_Maha1_norm = CorrectCount/40;
%% Mahalanobis Class 2

% Train model
[Maha, validate_Maha] = trainClassifier_Maha(train_validate(1:118,:),cov_ClassTwo);

yfit_Maha = Maha.predictFcn(test(:,2:14));
accuracy_Maha2 = 1-(nnz(yfit_Maha - test(:,1))/size(yfit_Maha - test(:,1),1));

%% Mahalanobis Class 3

% Train model
[Maha, validate_Maha] = trainClassifier_Maha(train_validate(1:118,:),cov_ClassThree);

yfit_Maha = Maha.predictFcn(test(:,2:14));
accuracy_Maha3 = 1-(nnz(yfit_Maha - test(:,1))/size(yfit_Maha - test(:,1),1));

%% Mahalanobis - normalised

% Train model
[Maha_norm, validate_Maha_norm] = trainClassifier_Maha(train_validate_norm(1:118,:),cov_All_norm);

yfit_Maha_norm = Maha_norm.predictFcn(test_norm(:,2:14));
accuracy_Maha_norm = 1-(nnz(yfit_Maha_norm - test(:,1))/size(yfit_Maha_norm - test(:,1),1));
%% Mahalanobis - normalised class1

% Train model
[Maha_norm, validate_Maha_norm] = trainClassifier_Maha(train_validate_norm(1:118,:),cov_ClassOne_norm);

yfit_Maha_norm = Maha_norm.predictFcn(test_norm(:,2:14));
accuracy_Maha_norm1 = 1-(nnz(yfit_Maha_norm - test(:,1))/size(yfit_Maha_norm - test(:,1),1));

%% Mahalanobis - normalised class2

% Train model
[Maha_norm, validate_Maha_norm] = trainClassifier_Maha(train_validate_norm(1:118,:),cov_ClassTwo_norm);

yfit_Maha_norm = Maha_norm.predictFcn(test_norm(:,2:14));
accuracy_Maha_norm2 = 1-(nnz(yfit_Maha_norm - test(:,1))/size(yfit_Maha_norm - test(:,1),1));

%% Mahalanobis - normalised class3

% Train model
[Maha_norm, validate_Maha_norm] = trainClassifier_Maha(train_validate_norm(1:118,:),cov_ClassThree_norm);

yfit_Maha_norm = Maha_norm.predictFcn(test_norm(:,2:14));
accuracy_Maha_norm3 = 1-(nnz(yfit_Maha_norm - test(:,1))/size(yfit_Maha_norm - test(:,1),1));

%% bar graphs for visualizing error rates of different metrics. 


%% K-means*******************************

%% normalised
% 'sqeuclidean'
k = 3;
% avoid local minima
[idx_sqeuclidean,cent_sqeuclidean,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','sqeuclidean','Display','final','Replicates',100);
sum(sumdist);

%visualise using Silhouette plot
figure;
[silh_sqeuclidean,h] = silhouette(train_validate_norm(1:118,2:14),idx_sqeuclidean,'sqeuclidean');
h = gca;
h.Children.EdgeColor = [.1 .1 1];
xlabel 'Silhouette Value'
ylabel 'Cluster'
title('K-means: sqeuclidean');

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

    PredictedClass = knnsearch(cent_sqeuclidean,TestData_norm_convert(:,2:14),'k',1,'Distance','euclidean'); 
%     PredictedClass = knnsearch(cent_sqeuclidean,TestData_norm_convert(:,2:14),'Distance','euclidean');
    if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_sqeuclidean = CorrectCount/size(test_norm,1);




%%
% cityblock
% avoid local minima
k = 3;
[idx_cityblock,cent_cityblock,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','cityblock','Display','final','Replicates',1000);
sum(sumdist);

%visualise using Silhouette plot
figure;
[silh_cityblock,h] = silhouette(train_validate_norm(1:118,2:14),idx_cityblock,'cityblock');
h = gca;
h.Children.EdgeColor = [.1 .1 1];
xlabel 'Silhouette Value'
ylabel 'Cluster'
title('K-means: cityblock');

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
accuracy_kmeans_cityblock = CorrectCount/size(test_norm,1);


%% normalised
% 'cosine'
% avoid local minima
k = 3;
[idx_cosine,cent_cosine,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','cosine','Display','final','Replicates',1000);
sum(sumdist);

%visualise using Silhouette plot
figure;
[silh_cosine,h] = silhouette(train_validate_norm(1:118,2:14),idx_cosine,'cosine');
h = gca;
h.Children.EdgeColor = [.1 .1 1];
xlabel 'Silhouette Value'
ylabel 'Cluster'
title('K-means: cosine');

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
for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
    PredictedClass = knnsearch(cent_cosine,TestData_norm_convert(:,2:14),'k',1,'Distance','cosine');
    if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_cosine = CorrectCount/size(test_norm,1);




%% normalised
% 'correlation' . best:0.7750, worst 0.65
% avoid local minima
k = 3;
[idx_correlation,cent_correlation,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','correlation','Display','final','Replicates',100);
sum(sumdist);

%visualise using Silhouette plot
figure;
[silh_correlation,h] = silhouette(train_validate_norm(1:118,2:14),idx_correlation,'correlation');
h = gca;
h.Children.EdgeColor = [.1 .1 1];
xlabel 'Silhouette Value'
ylabel 'Cluster'
title('K-means: correlation');

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
for i=1:1:size(test_norm,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
    PredictedClass = knnsearch(cent_correlation,TestData_norm_convert(:,2:14),'k',1,'Distance','correlation');
    if(PredictedClass(i) == TestData_norm_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_correlation = CorrectCount/size(test_norm,1);


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
accuracy_kmeans_maha = CorrectCount/size(test_norm,1);

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
accuracy_kmeans_maha1 = CorrectCount/size(test_norm,1);

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
accuracy_kmeans_maha2 = CorrectCount/size(test_norm,1);

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
accuracy_kmeans_maha3 = CorrectCount/size(test_norm,1);
