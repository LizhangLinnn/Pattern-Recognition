%% not normalised
% 'cosine'
% avoid local minima
k = 3;
[idx_cosine,cent_cosine,sumdist] = kmeans(TrainingData(1:118,2:14), k, 'Distance','cosine','Display','final','Replicates',1000);
sum(sumdist);

%visualise using Silhouette plot
figure;
[silh_cosine,h] = silhouette(TrainingData(1:118,2:14),idx_cosine,'cosine');
h = gca;
h.Children.EdgeColor = [.1 .1 1];
xlabel 'Silhouette Value'
ylabel 'Cluster'
title('K-means: cosine');

% convert test data indexes to the label we assigned in this case
TestData_convert = TestData;
for i=1:1:size(TestData_convert,1)
    if(TestData_convert(i,1) == 1)
        TestData_convert(i,1) = mode(idx_cosine(1:39));
    else if(TestData_convert(i,1) == 3)
        TestData_convert(i,1) = mode(idx_cosine(87:118));
        else
        TestData_convert(i,1) = mode(idx_cosine(40:86));
        end
    end
end

% input test data
CorrectCount = 0;
for i=1:1:size(TestData,1)
% % find the nn in the training data
%     for j=1:1:size(cent_cityblock,1)
%         for dim=2:1:size(TestData_norm,2)
%             % distance between the test data and the jth kmeans centre
%             d_cityblock(i,j) = (TestData_norm_convert(i,dim)-cent_cityblock(j,dim-1))^2 + d_cityblock(i);
%         end
%     end
    PredictedClass = knnsearch(cent_cosine,TestData_convert(:,2:14),'Distance','cosine');
    if(PredictedClass(i) == TestData_convert(i,1))
       CorrectCount = CorrectCount+1; 
    end
end

% accuracy
accuracy_kmeans_cosine_nonnorm = CorrectCount/size(TestData,1);
