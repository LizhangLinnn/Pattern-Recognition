%K means
loop = 100;
    k = 3;
accuracy_kmeans_sqeuclidean =zeros(loop,1);
accuracy_kmeans_cityblock =zeros(loop,1);
accuracy_kmeans_correlation =zeros(loop,1);
accuracy_kmeans_cosine =zeros(loop,1);
accuracy_kmeans_maha = zeros(loop,1);
accuracy_kmeans_maha1 = zeros(loop,1);
accuracy_kmeans_maha2 = zeros(loop,1);
accuracy_kmeans_maha3 = zeros(loop,1);

error_kmeans_sqeuclidean =zeros(loop,1);
error_kmeans_cityblock =zeros(loop,1);
error_kmeans_correlation =zeros(loop,1);
error_kmeans_cosine =zeros(loop,1);
error_kmeans_maha = zeros(loop,1);
error_kmeans_maha1 = zeros(loop,1);
error_kmeans_maha2 = zeros(loop,1);
error_kmeans_maha3 = zeros(loop,1);

fault = zeros(loop,1);
for n=1:loop
    load wine.data.csv
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


    train_validate_norm = normc(wine_data(:,2:14));
    train_validate_norm = horzcat(train_validate(:,1),vertcat(train_validate_norm([ClassOneRand(1:39)...
        ,ClassTwoRand(1:47),ClassThreeRand(1:32)],:),(train_validate_norm([ClassOneRand(40:46)...
        ,ClassTwoRand(48:55),ClassThreeRand(33:37)],:))));
    test_norm = normc(wine_data(:,2:14));
    test_norm = horzcat(test(:,1),test_norm([ClassOneRand(47:59),ClassTwoRand(56:71),ClassThreeRand(38:48)],:));


    % estimate covariance matrix (All features from all classes)
    % A.a
    cov_All = cov(train_validate(1:118,2:14));
%     cov_All = cov(train_validate(:,2:14));
    mean_All = mean(train_validate(1:118,2:14));

    % A.b
    cov_All_norm = cov(train_validate_norm(1:118,2:14));
%     cov_All_norm = cov(train_validate_norm(:,2:14));
    mean_All_norm = mean(train_validate_norm(1:118,2:14));

    % estimate covariance matrix (independently from class 1, 2 and 3)
    % A.a
    
%     idx_ClassOne = horzcat((1:39),(119:125));
    idx_ClassOne = horzcat((1:39));
    cov_ClassOne = cov(train_validate(idx_ClassOne,2:14));
    mean_ClassOne = mean(train_validate(idx_ClassOne,2:14));

%     idx_ClassTwo = horzcat((40:86),(126:133));
    idx_ClassTwo = horzcat((40:86));
    cov_ClassTwo = cov(train_validate(idx_ClassTwo,2:14));
    mean_ClassTwo = mean(train_validate(idx_ClassTwo,2:14));

%     idx_ClassThree = horzcat((87:118),(134:138));
    idx_ClassThree = horzcat((87:118));
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

    %class 1 -1:39, class 2 - 40:86, class 3 - 87:118
    %% normalised
    % 'sqeuclidean'
% if(fault(n)~=1)
    % avoid local minima
    [idx_sqeuclidean,cent_sqeuclidean,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','sqeuclidean','Display','final','Replicates',100);
    sum(sumdist);

    %visualise using Silhouette plot
%     figure;
%     [silh_sqeuclidean,h] = silhouette(train_validate_norm(1:118,2:14),idx_sqeuclidean,'sqeuclidean');
%     h = gca;
%     h.Children.EdgeColor = [.1 .1 1];
%     xlabel 'Silhouette Value'
%     ylabel 'Cluster'
%     title('K-means: sqeuclidean');

    [class1,class2,class3]=K_ConvertLabel_K(idx_sqeuclidean,k);

    
    %error of clustering
    idx_maha_convert = zeros(118,1);
    for i=1:118
        for j=1:size(class1,1)
            if(idx_sqeuclidean(i) == class1(j))
               idx_maha_convert(i) = 1; 
               break
            end
        end
        for j=1:size(class2,1)
            if(idx_sqeuclidean(i) == class2(j))
               idx_maha_convert(i) = 2; 
               break
            end
        end
        for j=1:size(class3,1)
            if(idx_sqeuclidean(i) == class3(j))
               idx_maha_convert(i) = 3; 
               break
            end
        end
    end
    count = 0;
    for i=1:118
        if(i<=39)
            if( idx_maha_convert(i) == 1)
                count=count+1;
            end
        elseif(i<=86)
            if( idx_maha_convert(i) == 2)
                count=count+1;
            end
        else
            if( idx_maha_convert(i) == 3)
                count=count+1;
            end
        end
    end
    error_kmeans_sqeuclidean(n) = 100 - count/118*100;
    
    % input test data
    PredictedClass = zeros(size(test_norm,1),1);
    CorrectCount = 0;
    PredictedClass = knnsearch(cent_sqeuclidean,test_norm(:,2:14),'k',1,'Distance','euclidean'); 

    for i=1:13
        for j=1:size(class1,1)
            if(PredictedClass(i) == class1(j))
                CorrectCount = CorrectCount + 1;
            end
        end
    end
    for i=14:29
        for j=1:size(class2,1)
            if(PredictedClass(i) == class2(j))
                CorrectCount = CorrectCount + 1;
            end
        end    
    end
    for i=30:40
        for j=1:size(class3,1)
            if(PredictedClass(i) == class3(j))
                CorrectCount = CorrectCount + 1;
            end
        end  
    end

    % accuracy
    accuracy_kmeans_sqeuclidean(n) = CorrectCount/size(test_norm,1);



% end
    %%
% if(fault(n) ~= 1)
    
    % cityblock
    % avoid local minima
    [idx_cityblock,cent_cityblock,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','cityblock','Display','final','Replicates',100);
    sum(sumdist);

    [class1,class2,class3]=K_ConvertLabel_K(idx_cityblock,k);
    
%     if(sum(class1(:,2))<0.7*39 || sum(class2(:,2))<0.7*47 || sum(class3(:,2))<0.7*32)
%         fault(n) = 1;
%     end

%     figure;
% silhouette(train_validate_norm(1:118,2:14),idx_cityblock,'cityblock');


    %error of clustering
    idx_maha_convert = zeros(118,1);
    for i=1:118
        for j=1:size(class1,1)
            if(idx_cityblock(i) == class1(j))
               idx_maha_convert(i) = 1; 
               break
            end
        end
        for j=1:size(class2,1)
            if(idx_cityblock(i) == class2(j))
               idx_maha_convert(i) = 2; 
               break
            end
        end
        for j=1:size(class3,1)
            if(idx_cityblock(i) == class3(j))
               idx_maha_convert(i) = 3; 
               break
            end
        end
    end
    count = 0;
    for i=1:118
        if(i<=39)
            if( idx_maha_convert(i) == 1)
                count=count+1;
            end
        elseif(i<=86)
            if( idx_maha_convert(i) == 2)
                count=count+1;
            end
        else
            if( idx_maha_convert(i) == 3)
                count=count+1;
            end
        end
    end
    error_kmeans_cityblock(n) = 100 - count/118*100;

    
    % input test data
    CorrectCount = 0;
    PredictedClass = knnsearch(cent_cityblock,test_norm(:,2:14),'k',1,'Distance','minkowski','p',1);

    for i=1:13
        for j=1:size(class1,1)
            if(PredictedClass(i) == class1(j))
                CorrectCount = CorrectCount + 1;
            end
        end
    end
    for i=14:29
        for j=1:size(class2,1)
            if(PredictedClass(i) == class2(j))
                CorrectCount = CorrectCount + 1;
            end
        end    
    end
    for i=30:40
        for j=1:size(class3,1)
            if(PredictedClass(i) == class3(j))
                CorrectCount = CorrectCount + 1;
            end
        end  
    end

    % accuracy
    accuracy_kmeans_cityblock(n) = CorrectCount/size(test_norm,1);
% end

    %% normalised
    % 'cosine'
    % avoid local minima
% if(fault(n) ~= 1)
    [idx_cosine,cent_cosine,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','cosine','Display','final','Replicates',100);
    sum(sumdist);

    [class1,class2,class3]=K_ConvertLabel_K(idx_cosine,k);

    %error of clustering
    idx_maha_convert = zeros(118,1);
    for i=1:118
        for j=1:size(class1,1)
            if(idx_cosine(i) == class1(j))
               idx_maha_convert(i) = 1; 
               break
            end
        end
        for j=1:size(class2,1)
            if(idx_cosine(i) == class2(j))
               idx_maha_convert(i) = 2; 
               break
            end
        end
        for j=1:size(class3,1)
            if(idx_cosine(i) == class3(j))
               idx_maha_convert(i) = 3; 
               break
            end
        end
    end
    count = 0;
    for i=1:118
        if(i<=39)
            if( idx_maha_convert(i) == 1)
                count=count+1;
            end
        elseif(i<=86)
            if( idx_maha_convert(i) == 2)
                count=count+1;
            end
        else
            if( idx_maha_convert(i) == 3)
                count=count+1;
            end
        end
    end
    error_kmeans_cosine(n) = 100 - count/118*100;

%% plot
%     figure;
% silhouette(train_validate_norm(1:118,2:14),idx_cosine,'cosine');
%%
    % input test data
    CorrectCount = 0;
    PredictedClass = knnsearch(cent_cosine,test_norm(:,2:14),'k',1,'Distance','cosine');

    for i=1:13
        for j=1:size(class1,1)
            if(PredictedClass(i) == class1(j))
                CorrectCount = CorrectCount + 1;
            end
        end
    end
    for i=14:29
        for j=1:size(class2,1)
            if(PredictedClass(i) == class2(j))
                CorrectCount = CorrectCount + 1;
            end
        end    
    end
    for i=30:40
        for j=1:size(class3,1)
            if(PredictedClass(i) == class3(j))
                CorrectCount = CorrectCount + 1;
            end
        end  
    end

    % accuracy
    accuracy_kmeans_cosine(n) = CorrectCount/size(test_norm,1);


% end

    %% normalised
    % 'correlation' . best:0.7750, worst 0.65
    % avoid local minima
% if(fault(n) ~= 1)
    [idx_correlation,cent_correlation,sumdist] = kmeans(train_validate_norm(1:118,2:14), k, 'Distance','correlation','Display','final','Replicates',100);
    sum(sumdist);

    [class1,class2,class3]=K_ConvertLabel_K(idx_correlation,k);

    
    %error of clustering
    idx_maha_convert = zeros(118,1);
    for i=1:118
        for j=1:size(class1,1)
            if(idx_correlation(i) == class1(j))
               idx_maha_convert(i) = 1; 
               break
            end
        end
        for j=1:size(class2,1)
            if(idx_correlation(i) == class2(j))
               idx_maha_convert(i) = 2; 
               break
            end
        end
        for j=1:size(class3,1)
            if(idx_correlation(i) == class3(j))
               idx_maha_convert(i) = 3; 
               break
            end
        end
    end
    count = 0;
    for i=1:118
        if(i<=39)
            if( idx_maha_convert(i) == 1)
                count=count+1;
            end
        elseif(i<=86)
            if( idx_maha_convert(i) == 2)
                count=count+1;
            end
        else
            if( idx_maha_convert(i) == 3)
                count=count+1;
            end
        end
    end
    error_kmeans_correlation(n) = 100 - count/118*100;
    %% plot
%     figure;
% silhouette(train_validate_norm(1:118,2:14),idx_correlation,'correlation');
%%

    % input test data
    CorrectCount = 0;
    PredictedClass = knnsearch(cent_correlation,test_norm(:,2:14),'k',1,'Distance','correlation');

    for i=1:13
        for j=1:size(class1,1)
            if(PredictedClass(i) == class1(j))
                CorrectCount = CorrectCount + 1;
            end
        end
    end
    for i=14:29
        for j=1:size(class2,1)
            if(PredictedClass(i) == class2(j))
                CorrectCount = CorrectCount + 1;
            end
        end    
    end
    for i=30:40
        for j=1:size(class3,1)
            if(PredictedClass(i) == class3(j))
                CorrectCount = CorrectCount + 1;
            end
        end  
    end

    % accuracy
    accuracy_kmeans_correlation(n) = CorrectCount/size(test_norm,1);
% end

    %% normalised
    % 'Mahalanobis' .
    % avoid local minima
% if(fault(n) ~= 1)
    [eig_vec, eig_val] = eig(inv(cov_All_norm));
    G = (((eig_val).^0.5)' * eig_vec');
    
    train_validate_norm_maha = (G*train_validate_norm(1:118,2:14)')';

    [idx_maha,cent_maha,sumdist] = kmeans(train_validate_norm_maha, k, 'Distance','sqeuclidean','Display','final','Replicates',100);
    sum(sumdist);

    %%
        %plot
% figure;
% silhouette(train_validate_norm_maha,idx_maha,'sqeuclidean');

%%
    [class1,class2,class3]=K_ConvertLabel_K(idx_maha,k);

    
    %error of clustering
    idx_maha_convert = zeros(118,1);
    for i=1:118
        for j=1:size(class1,1)
            if(idx_maha(i) == class1(j))
               idx_maha_convert(i) = 1; 
               break
            end
        end
        for j=1:size(class2,1)
            if(idx_maha(i) == class2(j))
               idx_maha_convert(i) = 2; 
               break
            end
        end
        for j=1:size(class3,1)
            if(idx_maha(i) == class3(j))
               idx_maha_convert(i) = 3; 
               break
            end
        end
    end
    count = 0;
    for i=1:118
        if(i<=39)
            if( idx_maha_convert(i) == 1)
                count=count+1;
            end
        elseif(i<=86)
            if( idx_maha_convert(i) == 2)
                count=count+1;
            end
        else
            if( idx_maha_convert(i) == 3)
                count=count+1;
            end
        end
    end
    error_kmeans_maha(n) = 100 - count/118*100;
    
    
    
    
    %knn
    CorrectCount = 0;
    PredictedClass = knnsearch(cent_maha,(G*test_norm(:,2:14)')','k',1,'Distance','euclidean');

    for i=1:13
        for j=1:size(class1,1)
            if(PredictedClass(i) == class1(j))
                CorrectCount = CorrectCount + 1;
            end
        end
    end
    for i=14:29
        for j=1:size(class2,1)
            if(PredictedClass(i) == class2(j))
                CorrectCount = CorrectCount + 1;
            end
        end    
    end
    for i=30:40
        for j=1:size(class3,1)
            if(PredictedClass(i) == class3(j))
                CorrectCount = CorrectCount + 1;
            end
        end  
    end

    % accuracy
    accuracy_kmeans_maha(n) = CorrectCount/size(test_norm,1);
% end
    %% normalised
    % 'Mahalanobis CLass one cov' .
    % avoid local minima
    
% if(fault(n) ~= 1)
        
    [eig_vec, eig_val] = eig(inv(cov_ClassOne_norm));
    G = (((eig_val).^0.5)' * eig_vec');

    train_validate_norm_maha = (G*train_validate_norm(1:118,2:14)')';

    [idx_maha,cent_maha,sumdist] = kmeans(train_validate_norm_maha, k, 'Distance','sqeuclidean','Display','final','Replicates',100);
    sum(sumdist);

        %%
% %         %plot
% figure;
% silhouette(train_validate_norm_maha,idx_maha,'sqeuclidean');

%%
    [class1,class2,class3]=K_ConvertLabel_K(idx_maha,k);
    
    
    
    %error of clustering    
    idx_maha_convert = zeros(118,1);
    for i=1:118
        for j=1:size(class1,1)
            if(idx_maha(i) == class1(j))
               idx_maha_convert(i) = 1; 
               break
            end
        end
        for j=1:size(class2,1)
            if(idx_maha(i) == class2(j))
               idx_maha_convert(i) = 2; 
               break
            end
        end
        for j=1:size(class3,1)
            if(idx_maha(i) == class3(j))
               idx_maha_convert(i) = 3; 
               break
            end
        end
    end
    count = 0;
    for i=1:118
        if(i<=39)
            if( idx_maha_convert(i) == 1)
                count=count+1;
            end
        elseif(i<=86)
            if( idx_maha_convert(i) == 2)
                count=count+1;
            end
        else
            if( idx_maha_convert(i) == 3)
                count=count+1;
            end
        end
    end
    error_kmeans_maha1(n) = 100 - count/118*100;
        
    % input test data
    CorrectCount = 0;
    PredictedClass = knnsearch(cent_maha,(G*test_norm(:,2:14)')','k',1,'Distance','euclidean');

    for i=1:13
        for j=1:size(class1,1)
            if(PredictedClass(i) == class1(j))
                CorrectCount = CorrectCount + 1;
            end
        end
    end
    for i=14:29
        for j=1:size(class2,1)
            if(PredictedClass(i) == class2(j))
                CorrectCount = CorrectCount + 1;
            end
        end    
    end
    for i=30:40
        for j=1:size(class3,1)
            if(PredictedClass(i) == class3(j))
                CorrectCount = CorrectCount + 1;
            end
        end  
    end
    % accuracy
    accuracy_kmeans_maha1(n) = CorrectCount/size(test_norm,1);
% end
    %% normalised
    % 'Mahalanobis' 2.
    % avoid local minima
% if(fault(n) ~= 1)
    [eig_vec, eig_val] = eig(inv(cov_ClassTwo_norm));
    G = (((eig_val).^0.5)' * eig_vec');

    train_validate_norm_maha = (G*train_validate_norm(1:118,2:14)')';

    [idx_maha,cent_maha,sumdist] = kmeans(train_validate_norm_maha, k, 'Distance','sqeuclidean','Display','final','Replicates',100);
    sum(sumdist);

        %%
        %plot
% figure;
% silhouette(train_validate_norm_maha,idx_maha,'sqeuclidean');

%%
    [class1,class2,class3]=K_ConvertLabel_K(idx_maha,k);
    
    
        %error of clustering
    idx_maha_convert = zeros(118,1);
    for i=1:118
        for j=1:size(class1,1)
            if(idx_maha(i) == class1(j))
               idx_maha_convert(i) = 1; 
               break
            end
        end
        for j=1:size(class2,1)
            if(idx_maha(i) == class2(j))
               idx_maha_convert(i) = 2; 
               break
            end
        end
        for j=1:size(class3,1)
            if(idx_maha(i) == class3(j))
               idx_maha_convert(i) = 3; 
               break
            end
        end
    end
    count = 0;
    for i=1:118
        if(i<=39)
            if( idx_maha_convert(i) == 1)
                count=count+1;
            end
        elseif(i<=86)
            if( idx_maha_convert(i) == 2)
                count=count+1;
            end
        else
            if( idx_maha_convert(i) == 3)
                count=count+1;
            end
        end
    end
    error_kmeans_maha2(n) = 100 - count/118*100;
    
    % input test data
    CorrectCount = 0;
    PredictedClass = knnsearch(cent_maha,(G*test_norm(:,2:14)')','k',1,'Distance','euclidean');

    for i=1:13
        for j=1:size(class1,1)
            if(PredictedClass(i) == class1(j))
                CorrectCount = CorrectCount + 1;
            end
        end
    end
    for i=14:29
        for j=1:size(class2,1)
            if(PredictedClass(i) == class2(j))
                CorrectCount = CorrectCount + 1;
            end
        end    
    end
    for i=30:40
        for j=1:size(class3,1)
            if(PredictedClass(i) == class3(j))
                CorrectCount = CorrectCount + 1;
            end
        end  
    end

    % accuracy
    accuracy_kmeans_maha2(n) = CorrectCount/size(test_norm,1);
% end
    %% normalised
    % 'Mahalanobis 3' .
    % avoid local minima
% if(fault(n) ~= 1)
    [eig_vec, eig_val] = eig(inv(cov_ClassThree_norm));
    G = (((eig_val).^0.5)'* eig_vec');

    train_validate_norm_maha = (G*train_validate_norm(1:118,2:14)')';

    [idx_maha,cent_maha,sumdist] = kmeans(train_validate_norm_maha, k, 'Distance','sqeuclidean','Display','final','Replicates',100);
    sum(sumdist);

        %%
        %plot
% figure;
% silhouette(train_validate_norm_maha,idx_maha,'sqeuclidean');
% 
%%
    [class1,class2,class3]=K_ConvertLabel_K(idx_maha,k);
    
    
    %error of clustering
    idx_maha_convert = zeros(118,1);
    for i=1:118
        for j=1:size(class1,1)
            if(idx_maha(i) == class1(j))
               idx_maha_convert(i) = 1; 
               break
            end
        end
        for j=1:size(class2,1)
            if(idx_maha(i) == class2(j))
               idx_maha_convert(i) = 2; 
               break
            end
        end
        for j=1:size(class3,1)
            if(idx_maha(i) == class3(j))
               idx_maha_convert(i) = 3; 
               break
            end
        end
    end
    count = 0;
    for i=1:118
        if(i<=39)
            if( idx_maha_convert(i) == 1)
                count=count+1;
            end
        elseif(i<=86)
            if( idx_maha_convert(i) == 2)
                count=count+1;
            end
        else
            if( idx_maha_convert(i) == 3)
                count=count+1;
            end
        end
    end
    error_kmeans_maha3(n) = 100 - count/118*100;
    
    % input test data
    CorrectCount = 0;
    PredictedClass = knnsearch(cent_maha,(G*test_norm(:,2:14)')','k',1,'Distance','euclidean');

    for i=1:13
        for j=1:size(class1,1)
            if(PredictedClass(i) == class1(j))
                CorrectCount = CorrectCount + 1;
            end
        end
    end
    for i=14:29
        for j=1:size(class2,1)
            if(PredictedClass(i) == class2(j))
                CorrectCount = CorrectCount + 1;
            end
        end    
    end
    for i=30:40
        for j=1:size(class3,1)
            if(PredictedClass(i) == class3(j))
                CorrectCount = CorrectCount + 1;
            end
        end  
    end
    % accuracy
    accuracy_kmeans_maha3(n) = CorrectCount/size(test_norm,1);
    
% end
end
% count = 0;
% select = [];
% for i = 1:loop
%    if(fault(i) == 0)
%       select(i-count) = i;
%    else
%        count = count+1;
%    end
% end

% accuracy_kmeans_sqeuclidean = accuracy_kmeans_sqeuclidean(select,:);
% accuracy_kmeans_cityblock = accuracy_kmeans_cityblock(select,:);
% accuracy_kmeans_correlation = accuracy_kmeans_correlation(select,:);
% accuracy_kmeans_cosine = accuracy_kmeans_cosine(select,:);
% accuracy_kmeans_maha = accuracy_kmeans_maha(select,:);
% accuracy_kmeans_maha1 = accuracy_kmeans_maha1(select,:);
% accuracy_kmeans_maha2 = accuracy_kmeans_maha2(select,:);
% accuracy_kmeans_maha3 = accuracy_kmeans_maha3(select,:);

std_kmeans_sqeuclidean =std(accuracy_kmeans_sqeuclidean);
std_kmeans_cityblock =std(accuracy_kmeans_cityblock);
std_kmeans_correlation =std(accuracy_kmeans_correlation);
std_kmeans_cosine =std(accuracy_kmeans_cosine);
std_kmeans_maha = std(accuracy_kmeans_maha);
std_kmeans_maha1 = std(accuracy_kmeans_maha1);
std_kmeans_maha2 = std(accuracy_kmeans_maha2);
std_kmeans_maha3 = std(accuracy_kmeans_maha3);


accuracy_kmeans_sqeuclidean =mean(accuracy_kmeans_sqeuclidean);
accuracy_kmeans_cityblock =mean(accuracy_kmeans_cityblock);
accuracy_kmeans_correlation =mean(accuracy_kmeans_correlation);
accuracy_kmeans_cosine =mean(accuracy_kmeans_cosine);
accuracy_kmeans_maha = mean(accuracy_kmeans_maha);
accuracy_kmeans_maha1 = mean(accuracy_kmeans_maha1);
accuracy_kmeans_maha2 = mean(accuracy_kmeans_maha2);
accuracy_kmeans_maha3 = mean(accuracy_kmeans_maha3);

error_kmeans_sqeuclidean =mean(error_kmeans_sqeuclidean);
error_kmeans_cityblock =mean(error_kmeans_cityblock);
error_kmeans_correlation =mean(error_kmeans_correlation);
error_kmeans_cosine =mean(error_kmeans_cosine);
error_kmeans_maha = mean(error_kmeans_maha);
error_kmeans_maha1 = mean(error_kmeans_maha1);
error_kmeans_maha2 = mean(error_kmeans_maha2);
error_kmeans_maha3 = mean(error_kmeans_maha3);


% accuracy_kmeans_sqeuclidean =median(accuracy_kmeans_sqeuclidean);
% accuracy_kmeans_cityblock =median(accuracy_kmeans_cityblock);
% accuracy_kmeans_correlation =median(accuracy_kmeans_correlation);
% accuracy_kmeans_cosine =median(accuracy_kmeans_cosine);
% accuracy_kmeans_maha = median(accuracy_kmeans_maha);
% accuracy_kmeans_maha1 = median(accuracy_kmeans_maha1);
% accuracy_kmeans_maha2 = median(accuracy_kmeans_maha2);
% accuracy_kmeans_maha3 = median(accuracy_kmeans_maha3);

%{
%% k=20
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9445;1-0.9408;1-0.9375;1-0.9255;1-0.9332;1-0.9340;1-0.9343;1-0.9352]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=10)');
xlabel('metrics');
ylabel('error rate / %');
%% k=10
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9283;1-0.9305;1-0.9105;1-0.9158;1-0.8990;1-0.9038;1-0.9110;1-0.9025]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=10)');
xlabel('metrics');
ylabel('error rate / %');

%% k=4
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9092;1-0.9075;1-0.8895;1-0.8830;1-0.9315;1-0.9330;1-0.9312;1-0.9092]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=10)');
xlabel('metrics');
ylabel('error rate / %');
%% k=5
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9085;1-0.9045;1-0.9067;1-0.9122;1-0.9010;1-0.9078;1-0.9137;1-0.9175]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=10)');
xlabel('metrics');
ylabel('error rate / %');
%% k=6
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9195;1-0.9205;1-0.9108;1-0.9038;1-0.8825;1-0.9020;1-0.9090;1-0.9047]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=10)');
xlabel('metrics');
ylabel('error rate / %');

%% after modification on maha
%% k=3 
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9238;1-0.9335;1-0.8898;1-0.8900;1-0.9082;1-0.8265;1-0.9750;1-0.9205]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=3)');
xlabel('metrics');
ylabel('error rate / %');

%% k=10
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9125;1-0.9293;1-0.9108;1-0.9103;1-0.8313;1-0.9630;1-0.9750;1-0.9467]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=10)');
xlabel('metrics');
ylabel('error rate / %');

%% k=6
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.8743;1-0.9190;1-0.9150;1-0.9025;1-0.8688;1-0.9402;1-0.9500;1-0.9312]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=10)');
xlabel('metrics');
ylabel('error rate / %');

%% k=14 not accurate
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9308;1-0.9370;1-0.9265;1-0.9307;1-0.8288;1-0.9645;1-0.9750;1-0.9530]);
set(gca,'xticklabel',name);
title('normalised data - K means clustering with different distance metrics (K=10)');
xlabel('metrics');
ylabel('error rate / %');

%}
%% 27-12-2016
%% k=3 
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9248;1-0.9397;1-0.8947;1-0.8952;1-0.8750;1-0.8255;1-0.9303;1-0.9280]);
set(gca,'xticklabel',name);
title('error rates of finding nearest cluster with diffrent distance metrics (K=3)');
xlabel('metrics');
ylabel('error rate / %');
set(gca,'fontsize',24);

figure;
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',[0.0353;0.0318;0.0424;0.0415;0.1158;0.1032;0.0382;0.0368]);
set(gca,'xticklabel',name);
title('standard deviation of error rates of finding nearest cluster with different metrics (K=3)');
xlabel('metrics');
ylabel('Std');
set(gca,'fontsize',24);

figure;
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',[7.5;6.5593;10.6441;10.4831;12.5593;15.4915;6.0424;7.2881]);
set(gca,'xticklabel',name);
title('error rates of clustering with different distance metrics (K=3)');
xlabel('metrics');
ylabel('error rates (%)');
set(gca,'fontsize',24);

%% k=10

name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',100*[1-0.9023;1-0.9250;1-0.9120;1-0.9128;1-0.8415;1-0.9630;1-0.9652;1-0.9473]);
set(gca,'xticklabel',name);
title('error rates of finding nearest cluster with diffrent distance metrics (K=10)');
xlabel('metrics');
ylabel('error rate / %');
set(gca,'fontsize',24);

figure;
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',[0.0467;0.0396;0.0427;0.0385;0.0784;0.0292;0.0351;0.0371]);
set(gca,'xticklabel',name);
title('standard deviation of error rates of finding nearest cluster with different metrics (K=10)');
xlabel('metrics');
ylabel('Std');
set(gca,'fontsize',24);

figure;
name = {'SqEuclidean';...
    'Cityblock';...
    'Cosine';...
    'Correlation';...
    'Maha';...
    'Maha1';...
    'Maha2';...
    'Maha3'};
bar([1:8]',[7.6695;6.1949;7.7712;7.7542;15.5254;1.8305;1.6864;3.5847]);
set(gca,'xticklabel',name);
title('error rates of clustering with different distance metrics (K=10)');
xlabel('metrics');
ylabel('error rates (%)');
set(gca,'fontsize',24);