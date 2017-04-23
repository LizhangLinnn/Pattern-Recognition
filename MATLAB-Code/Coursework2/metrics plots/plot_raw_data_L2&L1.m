%% raw data
% %class one dim 1 & 2
figure;
dim1 = 5;
dim2 = 13;


dim1 = dim1 +1;
dim2 = dim2+1;

% ***************** incorrect test data *****************
% data = vertcat(train_validate,incorrect_L2);
% gscatter(data(:,dim1),data(:,dim2),data(:,1));

% ***************** training_ validation *****************
h1(1:3) = gscatter(train_validate(1:118,dim1),train_validate(1:118,dim2),train_validate(1:118,1));
hold on

% scatter(mean_ClassOne(dim1-1),mean_ClassOne(dim2-1),'s');
% scatter(mean_ClassTwo(dim1-1),mean_ClassTwo(dim2-1),'s');
% scatter(mean_ClassThree(dim1-1),mean_ClassThree(dim2-1),'s');

plot_cov_ClassOne = [cov_ClassOne(dim1-1,dim1-1),cov_ClassOne(dim1-1,dim2-1);...
    cov_ClassOne(dim2-1,dim1-1), cov_ClassOne(dim2-1,dim2-1)];
plot_cov_ClassTwo = [cov_ClassTwo(dim1-1,dim1-1),cov_ClassTwo(dim1-1,dim2-1);...
    cov_ClassTwo(dim2-1,dim1-1), cov_ClassTwo(dim2-1,dim2-1)];
plot_cov_ClassThree = [cov_ClassThree(dim1-1,dim1-1),cov_ClassThree(dim1-1,dim2-1);...
    cov_ClassThree(dim2-1,dim1-1), cov_ClassThree(dim2-1,dim2-1)];
 
h = error_ellipse(plot_cov_ClassOne,[mean_ClassOne(dim1-1),mean_ClassOne(dim2-1)],'conf',0.80);
h = error_ellipse(plot_cov_ClassTwo,[mean_ClassTwo(dim1-1),mean_ClassTwo(dim2-1)],'conf',0.80);
h = error_ellipse(plot_cov_ClassThree,[mean_ClassThree(dim1-1),mean_ClassThree(dim2-1)],'conf',0.80);

% 
% r=0.02;
% ang=0:0.01:2*pi; 
% xp=r*cos(ang);
% yp=r*sin(ang);
% plot(mean_ClassOne_norm(dim1-1)+xp,mean_ClassOne_norm(dim2-1)+yp);
% 
% 
% ang=0:0.01:2*pi; 
% xp=r*cos(ang);
% yp=r*sin(ang);
% plot(mean_ClassTwo_norm(dim1-1)+xp,mean_ClassTwo_norm(dim2-1)+yp);
% 
% 
% ang=0:0.01:2*pi; 
% xp=r*cos(ang);
% yp=r*sin(ang);
% plot(mean_ClassThree_norm(dim1-1)+xp,mean_ClassThree_norm(dim2-1)+yp);

%labels and titles
% axis equal;
axis tight;

% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% %L2
% %misclassified to class 3
% h1(4:6) = gscatter(incorrect_L2(:,dim1),incorrect_L2(:,dim2),incorrect_L2(:,1));% failure case no. 5
% %correctly predicted
% h1(7:9) = gscatter(correct_L2(:,dim1),correct_L2(:,dim2),correct_L2(:,1)); %success case no. 1
% % zoom in
% limits = [1.88, 3.26,390, 873];
% axis(limits)

%L1
%misclassified to class 3
h1(4:6) = gscatter(incorrect_L1(:,dim1),incorrect_L1(:,dim2),incorrect_L1(:,1));% failure case no. 5
%correctly predicted
h1(7:9) = gscatter(correct_L1(:,dim1),correct_L1(:,dim2),correct_L1(:,1)); %success case no. 1
%plot data that was not correctly predicted by L2 but correct predicted by L1
j=0;k=0;
L1_success_L2_fail = [];
for i=1:size(incorrect_L2)
    j=j+1;
    if(isequal(incorrect_L1(j,:),incorrect_L2(i,:))==0)
        j= j-1;
        k = k+1;
        L1_success_L2_fail(k,:) = incorrect_L2(i,:);
    end
end
h1(10) = scatter(L1_success_L2_fail(:,dim1),L1_success_L2_fail(:,dim2),L1_success_L2_fail(:,1)); %only class 2 & 3
set(h1(10),'marker','o');
set(h1(10),'sizedata',200);
set(h1(10),'linewidth',3);
% zoom in
% limits = [1.88, 3.26,390, 873];
% axis(limits)


set(h1(1:3),'markersize',8);
set(h1(4:6),'marker','x');
set(h1(4:6),'markersize',10);
set(h1(4:6),'linewidth',2);
set(h1(7:9),'marker','+');
set(h1(7:9),'markersize',10);
set(h1(7:9),'linewidth',2);

legend(h1(1:9),{'Class 1','Class 2','Class 3',...
    'Class 1 failure','Class 2 failure','Class 3 failure'...
    ,'Class 1 successful','Class 2 successful','Class 3 successful','Corrected by L1'});
  
xlabel(strcat('dimension ',int2str(dim1-1)));
ylabel(strcat('dimension ',int2str(dim2-1)));
title(strcat('L1 - Successful and Failure Test Data Points in Dimension ',int2str(dim1-1),' and ',int2str(dim2-1)));
hold off;