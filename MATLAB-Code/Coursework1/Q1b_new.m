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

%Demonstrate with 1st set
TestIdx=test(c,1);                    %Create index list for test set
TrainingIdx=training(c,1);            %Index list for training set
TestData=X(:,TestIdx);              
TrainData=X(:,TrainingIdx);
%%
%find mean face image for training data
mean_face = mean(TrainData,2); %return a column vector which is the mean of training data

%compute the covariance matrix
Phi_TestData = TestData - mean_face;  %Obtain test data by subtracting from mean face
Phi_TrainData = TrainData - mean_face; %Obtain train data
S = (Phi_TrainData' * Phi_TrainData)/size(Phi_TrainData,2); %A'A

%compute and normalise the eigenvectors of covariance matrix S
[eig_vec, eig_val] = eig(S); 
eig_vec = Phi_TrainData * eig_vec;
eig_vec = normc(eig_vec);
[eig_val_sort, eig_val_sort_index] = sort(diag(eig_val),'descend');% diag(eig_val) returns a vector 
                                                                   % that contains elements of diagonal matrix
%Show the magnitude of eigenvalues
figure;
subplot(1,2,1)
plot(eig_val_sort, 'LineWidth', 2.5)
xlabel('No. of eigenvalues');
ylabel('Magnetude of eigenvalues');
title('Eigenvalues in Decreasing Order');
subplot(1,2,2)
plot(eig_val_sort, 'LineWidth', 2.5)
axis([0 700 0 1300])
xlabel('No. of eigenvalues');
ylabel('Magnetude of eigenvalues');
title('Eigenvalues in Decreasing Order');

figure;
plot(eig_val_sort, 'LineWidth', 2.5)
axis([0 700 0 1300])
xlabel('No. of eigenvalues');
ylabel('Magnetude of eigenvalues');
title('Eigenvalues in Decreasing Order');

%plot the mean face
figure;
imagesc(reshape(mean_face,56,46));
title('mean face')
colormap gray

%Plot the three most significant eigenfaces
figure;
subplot(3,3,2)
imagesc(reshape(mean_face,56,46));
title('mean face')
subplot(3,3,4)
imagesc(reshape(eig_vec(:,eig_val_sort_index(1)),56,46));
title('Eigenface 1');
subplot(3,3,5)
imagesc(reshape(eig_vec(:,eig_val_sort_index(2)),56,46));
title('Eigenface 2');
subplot(3,3,6)
imagesc(reshape(eig_vec(:,eig_val_sort_index(3)),56,46));
title('Eigenface 3');
subplot(3,3,7)
imagesc(reshape(eig_vec(:,eig_val_sort_index(4)),56,46));
title('Eigenface 4');
subplot(3,3,8)
imagesc(reshape(eig_vec(:,eig_val_sort_index(466)),56,46));
title('Eigenface 466');
subplot(3,3,9)
imagesc(reshape(eig_vec(:,eig_val_sort_index(467)),56,46));
title('Eigenface 467');
colormap gray


%Find the suitable number of eigenvectors
%in PCA, feature information is represented by the amount of total variance
% thus, if we want to extract 95% of total information, the same amount of
% total covariance needs to be extracted from principle components (PC)
covariance=0;
total_covariance = sum(eig_val_sort);
%threshold = 0.92 ; 
threshold = 0:0.005:1;
number_of_PC = zeros(1,size(threshold,2));
for i = 1:1:size(threshold,2)
    covariance = 0;
    while covariance < threshold(i) * total_covariance
        number_of_PC(i) = number_of_PC(i) + 1;
        covariance = covariance + eig_val_sort(number_of_PC(i));
    end
end
figure;
plot(number_of_PC, threshold*100);
title('% of Total Covariance against Choice of Number of Principle Components');


%% keep the best M eigenvectors
M_range = [468 150 20]; %[468 150 50]

%{
%% by varying M, compare the reconstructed training image with the original one
figure;
i = [1 100 250]; %choose image from train data
for j = 1:1:size(i,2)
    %plot original
    subplot(3,4,4*(j-1)+1);
    imagesc(reshape(TrainData(:,i(j)),56,46));
    str = sprintf('%d th Training Image', i(j));
    title(str);

    
    
    for M_index=1:1:size(M_range,2)
        M = M_range(M_index);
               
            %compute the best M eigenvectors
            M_eig_vec = eig_vec(:, eig_val_sort_index(1:M));

            %projection of normalised training faces
            Projection_M_eig_vec = (Phi_TrainData' * M_eig_vec)';

            %face reconstruction using the best M eigenvectors
            x = mean_face + M_eig_vec * Projection_M_eig_vec;

            %compare the reconstructed image with the original one
            subplot(3,4,4*(j-1)+1+M_index)
            imagesc(reshape(x(:,i(j)),56,46));
            str = sprintf('Best %d Eigenvectors' , M);
            title(str);            
            
    end
end
colormap gray



%% by varying M, compare the reconstructed training image with the original one
figure;
i = [1 10 30]; %choose image from train data
for j = 1:1:size(i,2)
    %plot original
    subplot(3,4,4*(j-1)+1);
    imagesc(reshape(TestData(:,i(j)),56,46));
    str = sprintf('%d th Test Image', i(j));
    title(str);
    
    
    for M_index=1:1:size(M_range,2)
        M = M_range(M_index);
            %compute the best M eigenvectors
            M_eig_vec = eig_vec(:, eig_val_sort_index(1:M));

            %projection of normalised training faces
            Projection_M_eig_vec = (Phi_TestData' * M_eig_vec)';

            %face reconstruction using the best M eigenvectors
            x = mean_face + M_eig_vec * Projection_M_eig_vec;

            %compare the reconstructed image with the original one
            subplot(3,4,4*(j-1)+M_index+1)
            imagesc(reshape(x(:,i(j)),56,46));
            str = sprintf('Best %d Eigenvectors' , M);
            title(str);            
            
    end
end
colormap gray
%}

%{
%% 
% plot the relationship between reconstruction error for training image 
% and the number of eigenvectors used
figure;
ave_reconstruct_error_training = zeros(size(TrainData,2)-1,1);
for M =1:1:(size(TrainData,2)-1)
    %find the best M eigvector
    M_eig_vec = eig_vec(:,eig_val_sort_index(1:1:M));
    %reconstruction
    Projection_M_eig_vec = (Phi_TrainData' * M_eig_vec)';
    x = mean_face + M_eig_vec * Projection_M_eig_vec;
    %error
    ave_reconstruct_error_training(M,1) = sum((rms(x-TrainData)))/size(TrainData,2);
    
    
end
plot([1:1:(size(TrainData,2)-1)],ave_reconstruct_error_training);

%% 
% plot the relationship between reconstruction error for test image 
% and the number of eigenvectors used
hold on;
ave_reconstruct_error_test = zeros(size(TrainData,2)-1,1);
for M =1:1:(size(TrainData,2)-1)
    %find the best M eigvector
    M_eig_vec = eig_vec(:,eig_val_sort_index(1:1:M));
    %reconstruction
    Projection_M_eig_vec = (Phi_TestData' * M_eig_vec)';
    x = mean_face + M_eig_vec * Projection_M_eig_vec;
    %error
    ave_reconstruct_error_test(M,1) = sum((rms(x-TestData)))/size(TestData,2);
    
    
end
plot([1:1:(size(TrainData,2)-1)],ave_reconstruct_error_test);
xlabel('no. of eigenvectors');
ylabel('average reconstruction error');
title('reconstruction errors for training & test images vs the no. of eigenvectors');
legend('reconstruction error for training', 'reconstruction error for test');
%}


%% NN classification method
figure;
accuracy = zeros(10,size(TrainData,2));
elapsedTime = zeros(10,size(TrainData,2));
for j=1:10
TestIdx=test(c,j);                    %Create index list for test set
TrainingIdx=training(c,j);            %Index list for training set
TestData=X(:,TestIdx);              
TrainData=X(:,TrainingIdx);

%find mean face image for training data
mean_face = mean(TrainData,2); %return a column vector which is the mean of training data

%compute the covariance matrix
Phi_TestData = TestData - mean_face;  %Obtain test data by subtracting from mean face
Phi_TrainData = TrainData - mean_face; %Obtain train data
S = (Phi_TrainData' * Phi_TrainData)/size(Phi_TrainData,2); %A'A

%compute and normalise the eigenvectors of covariance matrix S
[eig_vec, eig_val] = eig(S); 
eig_vec = Phi_TrainData * eig_vec;
eig_vec = normc(eig_vec);
[eig_val_sort, eig_val_sort_index] = sort(diag(eig_val),'descend');% diag(eig_val) returns a vector 

    for M = 1:1:size(TrainData,2)-200
        %compute the best M eigenvectors
        tic;
        M_eig_vec = eig_vec(:, eig_val_sort_index(1:M));

        %projection of normalised training faces
        Projection_M_eig_vec = (Phi_TrainData' * M_eig_vec)';

        %normalise Phi_TestData
        %Phi_TestData_norm = normc(Phi_TestData);

        %project on the eigenspace
        Projection_TestData = (Phi_TestData' * M_eig_vec)';
        %error
        Correct_Match = 0;
        LB_PredictedData = zeros(size(TestData,2),1);
        for i=1:1:size(Projection_TestData,2)
            error = Projection_TestData(:,i) - Projection_M_eig_vec;

            %minimum error and its index
            [MinError, LB_PredictedData(i,1)] = min(rms(error)); %%NN methodindex
            if (LB_PredictedData(i,1) > (i-1)*9 && LB_PredictedData(i,1) <= i*9)
                Correct_Match = Correct_Match + 1;
            end
        end
        elapsedTime(j,M)=toc;

        accuracy(j,M) = Correct_Match/size(TestData, 2);

    end
end
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
alldatacursors = findall(gcf,'type','hggroup')
set(alldatacursors,'FontSize',18)
%% example success and failure cases

%success
figure;
subplot(121);
imagesc(reshape(TrainData(:,LB_PredictedData(1,1)),56,46));
title('Predicted Image');
subplot(122);
imagesc(reshape(TestData(:,1),56,46));
title('Actual Image');
colormap gray

%fail
figure;
subplot(121);
imagesc(reshape(TrainData(:,LB_PredictedData(2,1)),56,46));
title('Predicted Image');
subplot(122);
imagesc(reshape(TestData(:,2),56,46));
title('Actual Image');
colormap gray

mode

%% confusion matrix
% true positive
% faulse positive
% predicted data is correct if the image with minimum error lies in the
LB_PredictedData = ceil(LB_PredictedData/9);
LB_ActualData = [1:1:52];
Confusion_Matrix_NN = confusionmat(LB_ActualData,LB_PredictedData);

CMNN_HeatMap=HeatMap(Confusion_Matrix_NN,'Colormap', 'redgreencmap' ,'RowLabels',[1:52],'ColumnLabels',[1:52]);




%% time/memory


