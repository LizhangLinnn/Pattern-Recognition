function  [accuracy mismatch_Chi2] = Chi2(TrainData, TestData)
    %train
    
    mismatch_Chi2 = [];
    
    %find nearest neighbour
    Chi2_distance = zeros(size(TestData,1),size(TrainData,1));
    match = zeros(size(TestData,1),2);
    CountCorrect = 0; k=0;
    for i=1:1:size(TestData,1)
       for j=1:1:size(TrainData,1) 
           % calculate the Chi2 distance between the test data and the train
           % data
           for dim=2:1:14
%                Chi2_distance(i,j) = 0.5 * (TestData(i,dim)-TrainData(j,dim))^2/(TestData(i,dim)+TrainData(j,dim)) + Chi2_distance(i,j);
%                Chi2_distance(i,j) = chi_square_statistics(TestData(i,dim),TrainData(j,dim)) + Chi2_distance(i,j);
               Chi2_distance(i,j) = 0.5 * (TestData(i,dim)-TrainData(j,dim))*(TestData(i,dim)-TrainData(j,dim))/(TestData(i,dim)+TrainData(j,dim)) + Chi2_distance(i,j);

           end
       end
       
       [min_val matched_index] = min(Chi2_distance(i,:));
       %the label of actual result
       match(i,1) = TestData(i,1);
       %store the label of predicted result
       match(i,2) = TrainData(matched_index,1);
       if(TestData(i,1) == TrainData(matched_index,1))
          CountCorrect = CountCorrect+1; 
       else
           k=k+1;
           mismatch_Chi2(1,k) = i;
       end
    end
    
    % calculate accuracy
    accuracy = CountCorrect/size(TestData,1);
        
end