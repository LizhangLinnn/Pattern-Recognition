function  [accuracy,mismatch_hist] = histogram_intersection_norm(TrainData, TestData)
    %train
    
    %find nearest neighbour
    hist_distance = zeros(size(TestData,1),size(TrainData,1));
    match = zeros(size(TestData,1),2);
    CountCorrect = 0;k=0;
    mismatch_hist = [];
    for i=1:1:size(TestData,1)
       for j=1:1:size(TrainData,1) 
           % calculate the Chi2 distance between the test data and the train
           % data
           hist_distance(i,j) = histogram_intersection_d_norm(TestData(i,2:14),TrainData(j, 2:14));
       end
       
       [min_val matched_index] = min(hist_distance(i,:));
       %the label of actual result
       match(i,1) = TestData(i,1);
       %store the label of predicted result
       match(i,2) = TrainData(matched_index,1);
       if(TestData(i,1) == TrainData(matched_index,1))
          CountCorrect = CountCorrect+1; 
       else
           k=k+1;
           mismatch_hist(k) = i;
       end
    end
    
    % calculate accuracy
    accuracy = CountCorrect/size(TestData,1);
        
end