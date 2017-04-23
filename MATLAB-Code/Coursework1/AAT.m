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

%find mean face image for training data
mean_face = mean(TrainData,2); %return a column vector which is the mean of training data

%compute the covariance matrix
Phi_TestData = TestData - mean_face;  %Obtain test data by subtracting from mean face
Phi_TrainData = TrainData - mean_face; %Obtain train data
S = (Phi_TrainData * Phi_TrainData')/size(Phi_TrainData,2); %A'A

%compute and normalise the eigenvectors of covariance matrix S
[eig_vecAAT, eig_valAAT] = eig(S); 
eig_vecAAT = normc(eig_vecAAT);
[eig_val_sortAAT, eig_val_sort_indexAAT] = sort(diag(eig_valAAT),'descend');% diag(eig_val) returns a vector 
                                                                   % that contains elements of diagonal matrix

%plot the mean face
figure;
imagesc(reshape(mean_face,56,46));
title('mean face')
colormap gray

%Plot the three most significant eigenfaces
figure;
subplot(2,3,1)
imagesc(reshape(eig_vecAAT(:,eig_val_sort_indexAAT(1)),56,46));
title('Eigenface 1');
subplot(2,3,2)
imagesc(reshape(eig_vecAAT(:,eig_val_sort_indexAAT(2)),56,46));
title('Eigenface 2');
subplot(2,3,3)
imagesc(reshape(eig_vecAAT(:,eig_val_sort_indexAAT(3)),56,46));
title('Eigenface 3');
subplot(2,3,4)
imagesc(reshape(eig_vecAAT(:,eig_val_sort_indexAAT(4)),56,46));
title('Eigenface 4');
subplot(2,3,5)
imagesc(reshape(eig_vecAAT(:,eig_val_sort_indexAAT(467)),56,46));
title('Eigenface 467');
subplot(2,3,6)
imagesc(reshape(eig_vecAAT(:,eig_val_sort_indexAAT(468)),56,46));
title('Eigenface 468');
colormap gray