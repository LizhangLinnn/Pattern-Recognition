clc
clear
close all

%load face data
load face.mat

%% Q1

%10-fold crossvalidation
%10 items in each class and 9 data into training set, 1 into test set, same
%as leave-one-out in this case
k=10;                               %Define ratio of partition, k is the proportion sorted into test set
c = cvpartition(l,'Kfold',k);       %Create partition object

%Demonstrate with 1st set
TestIdx=test(c,1);                    %Create index list for test set
TrainingIdx=training(c,1);            %Index list for training set
TestDataTemp=X(:,TestIdx);              
TrainDataTemp=X(:,TrainingIdx);

%find mean face image for training data
mean_face = mean(TrainDataTemp,2); %return a column vector which is the mean of training data

%compute the covariance matrix
TestData = TestDataTemp - mean_face;  %Obtain test data by subtracting from mean face
Phi_TrainData = TrainDataTemp - mean_face; %Obtain train data
S = (Phi_TrainData * Phi_TrainData')/size(Phi_TrainData,2); %AA'

%compute and normalise the eigenvectors of covariance matrix S
[eig_vec, eig_val] = eig(S); 
eig_vec = normc(eig_vec);
[eig_val_sort, eig_val_sort_index] = sort(diag(eig_val),'descend');% diag(eig_val) returns a vector 
                                                                   % that contains elements of diagonal matrix
%Show the magnitude of eigenvalues
figure;
subplot(1,2,1)
plot(eig_val_sort)
xlabel('No. of eigenvalues');
ylabel('Magnetude of eigenvalues');
title('Eigenvalues in Decreasing Order');
subplot(1,2,2)
plot(eig_val_sort)
axis([0 700 0 1300])
xlabel('No. of eigenvalues');
ylabel('Magnetude of eigenvalues');
title('Eigenvalues in Decreasing Order');

figure;
plot(eig_val_sort)
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
% thus, if we want to extract 99% of total information, the same amount of
% total covariance needs to be extracted from principle components (PC)
covariance=0;
total_covariance = sum(eig_val_sort);
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

