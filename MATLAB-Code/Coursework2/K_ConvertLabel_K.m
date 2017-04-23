function [class1_cluster, class2_cluster, class3_cluster] = K_ConvertLabel_K(idx,k)

cluster = zeros(k,4);
for i=1:k
   cluster(i,1) = i; 
end

for i=1:118
    if(i<=39)
        cluster(idx(i),2)=cluster(idx(i),2)+1;
    elseif(i<=86)
        cluster(idx(i),3)=cluster(idx(i),3)+1;
    else
        cluster(idx(i),4)=cluster(idx(i),4)+1;
    end
end

index = zeros(k,1);
class1_cluster = [];
class2_cluster = [];
class3_cluster = [];
for i=1:k
    [~ , index(i)] = max(cluster(i,2:4));
    if(index(i)==1)
        class1_cluster = vertcat(class1_cluster,i);
    elseif(index(i)==2)
        class2_cluster = vertcat(class2_cluster,i);
    elseif(index(i)==3)
        class3_cluster = vertcat(class3_cluster,i);
    end
end
