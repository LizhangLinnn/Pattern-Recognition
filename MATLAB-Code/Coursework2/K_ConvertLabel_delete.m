function [count_class1, count_class2, count_class3] = K_ConvertLabel_delete(idx)
fault = 0; % if fault = 1, then regenerate data.
count_class1 = [];
count_class2 = [];
count_class3 = [];

for i=1:39
   if(i==1)
       count_class1(1,1:2) = [idx(i),1];
   else
       counted = 0;
       for j=1:size(count_class1,1)
          if(count_class1(j,1)==idx(i))
              count_class1(j,2)=count_class1(j,2)+1;
              counted = 1;
          end
       end
       if(counted == 0)
           count_class1 = vertcat(count_class1,[idx(i),1]);
       end
   end
end





for i=40:86
   if(i==40)
       count_class2(1,1:2) = [idx(i),1];
   else
       counted = 0;
       for j=1:size(count_class2,1)
          if(count_class2(j,1)==idx(i))
              count_class2(j,2)=count_class2(j,2)+1;
              counted = 1;
          end
       end
       if(counted == 0)
           count_class2 = vertcat(count_class2,[idx(i),1]);
       end
   end
end

for i=87:118
   if(i==87)
       count_class3(1,1:2) = [idx(i),1];
   else
       counted = 0;
       for j=1:size(count_class3,1)
          if(count_class3(j,1)==idx(i))
              count_class3(j,2)=count_class3(j,2)+1;
              counted = 1;
          end
       end
       if(counted == 0)
           count_class3 = vertcat(count_class3,[idx(i),1]);
       end
   end
end
%%
% convert test data indexes to the label we assigned in this case

count_class1_new = count_class1;
count_class2_new = count_class2;
count_class3_new = count_class3;
count = 0;
for i=1:size(count_class1,1)
    for j=1:size(count_class2,1)
        if(count_class1(i,1) == count_class2(j,1))
            if(count_class1(i,2) <count_class2(j,2))
                count = count+1;
               count_class1_new = count_class1_new([1:i-count,i-count+2:end],:);%delete the row in class 1 
               break
            end
        end
    end
end
count =0;
count_class1_new2 = count_class1_new;
for i=1:size(count_class1_new,1)
    for j=1:size(count_class3,1)
        if(count_class1_new(i,1) == count_class3(j,1))
            if(count_class1_new(i,2) < count_class3(j,2))
                count = count+1;
               count_class1_new2 = count_class1_new2([1:i-count,i-count+2:end],:);%delete the row in class 1 
               break
            end
        end
    end
end

count = 0;
for i=1:size(count_class2,1)
    for j=1:size(count_class1,1)
        if(count_class2(i,1) == count_class1(j,1))
            if(count_class2(i,2) <count_class1(j,2))
                count = count+1;
               count_class2_new = count_class2_new([1:i-count,i-count+2:end],:);%delete the row in class 1 
               break
            end
        end
    end
end
count = 0;
count_class2_new2 = count_class2_new;
for i=1:size(count_class2_new,1)
    for j=1:size(count_class3,1)
        if(count_class2_new(i,1) == count_class3(j,1))
            if(count_class2_new(i,2) <count_class3(j,2))
                count = count+1;
               count_class2_new2 = count_class2_new2([1:i-count,i-count+2:end],:);%delete the row in class 1 
               break
            end
        end
    end
end

count = 0;
for i=1:size(count_class3,1)
    for j=1:size(count_class1,1)
        if(count_class3(i,1) == count_class1(j,1))
            if(count_class3(i,2) <count_class1(j,2))
                count = count+1;
               count_class3_new = count_class3_new([1:i-count,i-count+2:end],:);%delete the row in class 1 
               break
            end
        end
    end
end

count = 0;
count_class3_new2 = count_class3_new;
for i=1:size(count_class3_new,1)
    for j=1:size(count_class2,1)
        if(count_class3_new(i,1) == count_class2(j,1))
            if(count_class3_new(i,2) <count_class2(j,2))
                count = count+1;
               count_class3_new2 = count_class3_new2([1:i-count,i-count+2:end],:);%delete the row in class 1 
               break
            end
        end
    end
end

%%
count_class1 = count_class1_new2; % old label matched to new labels
count_class2 = count_class2_new2; % old label matched to new labels
count_class3 = count_class3_new2; % old label matched to new labels
