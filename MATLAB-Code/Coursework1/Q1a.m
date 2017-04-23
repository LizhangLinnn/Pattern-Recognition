%partition face data into training and testing
load('face.mat');
fraction = 9/10;
NumOfTrainingData = 520*fraction;
NumOfTestingData = 520 - 520*fraction;
TrainingData = zeros(2576,NumOfTrainingData);
TestingData = zeros(2576,NumOfTestingData);

for i=1:1:(520/10)
    for j=1:1:10
        if(j<=fraction*10)
            TrainingData(:,(fraction*10*(i-1)+j)) = X(:,((i-1)*10+j));
        else
            TestingData(:,((10-fraction*10)*(i-1)+j-fraction*10)) = X(:,((i-1)*10+j));
        end
    end
end

%compute the average face vector for training data set
MeanFace = 0;
for i=1:1:NumOfTrainingData
    MeanFace = TrainingData(:,i)+ MeanFace;
end
MeanFace = MeanFace./520;

%Compute the covariance matrix S
A = TrainingData - MeanFace;
S = A*A' ./ NumOfTrainingData;

%compute the eigenvectors of covariance matrix S
[eig_vec, eig_val] = eig(S); 
%   This produces a diagonal matrix eig_val of eigenvalues and 
%   a full matrix eig_val whose columns are the corresponding eigenvectors  
%   so that S*eig_vec = eig_vec*eig_val.
eig_vec = normc(eig_vec);
figure;
[eig_val_sort, eig_val_sort_index] = sort(diag(eig_val),'descend');% diag(eig_val) returns a vector 
                                                                   % that contains elements of diagonal matrix
plot(eig_val_sort, 'LineWidth',2.5); 
xlabel('No. of eigenvalues');
ylabel('Magnetude of eigenvalues');
title('Eigenvalues in Decreasing Order');

%keep the best M eigenvectors
M = 465;
M_eig_vec = eig_vec(:,eig_val_sort_index(1:M)) %dim 2576*M

%normalised eigenfaces
NormTraining = normc(A'*M_eig_vec)';


%% Testing

%generate an unknown face image
UnknownFaceIndex = randi(52,1);
UnknownFace =  TestingData(:,UnknownFaceIndex);
figure;
imagesc(reshape(UnknownFace,56,46));
colormap gray;
%normalise the image
NormUnknownFace =  UnknownFace - MeanFace;
%project on the eigenspace
NormTesting = normc(NormUnknownFace' * M_eig_vec)';
%error
error = NormTesting - NormTraining;
%minimum error and its index
[MinError, index] = min(rms(error)); %%NN method
MatchFace = TrainingData(:,index);
figure;
imagesc(reshape(MatchFace,56,46));
colormap gray;

