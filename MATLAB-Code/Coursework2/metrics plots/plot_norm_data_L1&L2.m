%% normalised
% %class one dim 1 & 2
figure;
dim1 = 2;
dim2 = 10;


dim1 = dim1 +1;
dim2 = dim2+1;

% ***************** incorrect test data *****************
% data = vertcat(train_validate,incorrect_L2);
% gscatter(data(:,dim1),data(:,dim2),data(:,1));

% ***************** training_ validation *****************
% h1(1:3) = gscatter(train_validate(:,dim1),train_validate(:,dim2),train_validate(:,1));
h1(1:3) = gscatter(train_validate_norm(:,dim1),train_validate_norm(:,dim2),train_validate(:,1));
hold on

% scatter(mean_ClassOne_norm(dim1-1),mean_ClassOne_norm(dim2-1),'*');
% scatter(mean_ClassTwo_norm(dim1-1),mean_ClassTwo_norm(dim2-1),'*');
% scatter(mean_ClassThree_norm(dim1-1),mean_ClassThree_norm(dim2-1),'*');


plot_cov_ClassOne_norm = [cov_ClassOne_norm(dim1-1,dim1-1),cov_ClassOne_norm(dim1-1,dim2-1);...
    cov_ClassOne_norm(dim2-1,dim1-1), cov_ClassOne_norm(dim2-1,dim2-1)];
plot_cov_ClassTwo_norm = [cov_ClassTwo_norm(dim1-1,dim1-1),cov_ClassTwo_norm(dim1-1,dim2-1);...
    cov_ClassTwo_norm(dim2-1,dim1-1), cov_ClassTwo_norm(dim2-1,dim2-1)];
plot_cov_ClassThree_norm = [cov_ClassThree_norm(dim1-1,dim1-1),cov_ClassThree_norm(dim1-1,dim2-1);...
    cov_ClassThree_norm(dim2-1,dim1-1), cov_ClassThree_norm(dim2-1,dim2-1)];

h = error_ellipse(plot_cov_ClassOne_norm,[mean_ClassOne_norm(dim1-1),mean_ClassOne_norm(dim2-1)],'conf',0.9);
h = error_ellipse(plot_cov_ClassTwo_norm,[mean_ClassTwo_norm(dim1-1),mean_ClassTwo_norm(dim2-1)],'conf',0.9);
h = error_ellipse(plot_cov_ClassThree_norm,[mean_ClassThree_norm(dim1-1),mean_ClassThree_norm(dim2-1)],'conf',0.9);

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
axis equal;
axis tight;

% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
% ******************plot test ***********************************
%L1
% Three class two test data misclassified
h1(4) = scatter(incorrect_L1_norm(:,dim1),incorrect_L1_norm(:,dim2),incorrect_L1_norm(:,1),'g');% failure case no. 5

% NN training data
h1(5) = scatter(train_validate_norm([35,94,27], dim1),train_validate_norm([35,94,27], dim2));

n=100; %number of points that form a line
input = 35;input2 =1;
    d = (incorrect_L1_norm(input2,dim1)-train_validate_norm(input,dim1))/n;
    x1= train_validate_norm(input,dim1):d:incorrect_L1_norm(input2,dim1);
    if(d==0)
        x1 = repmat(train_validate_norm(input,dim1),[1, n+1]);
    end
    d = (incorrect_L1_norm(input2,dim2)-train_validate_norm(input,dim2))/n;
    x2= train_validate_norm(input,dim2):d:incorrect_L1_norm(input2,dim2);
    if(d==0)
        x2 = repmat(train_validate_norm(input,dim2),[1, n+1]);
    end
scatter(x1,x2,4);


input = 94;input2 =2;
    d = (incorrect_L1_norm(input2,dim1)-train_validate_norm(input,dim1))/n;
    x1= train_validate_norm(input,dim1):d:incorrect_L1_norm(input2,dim1);
    if(d==0)
        x1 = repmat(train_validate_norm(input,dim1),[1, n+1]);
    end
    d = (incorrect_L1_norm(input2,dim2)-train_validate_norm(input,dim2))/n;
    x2= train_validate_norm(input,dim2):d:incorrect_L1_norm(input2,dim2);
    if(d==0)
        x2 = repmat(train_validate_norm(input,dim2),[1, n+1]);
    end
scatter(x1,x2,4);

input = 27;input2 =3;
    d = (incorrect_L1_norm(input2,dim1)-train_validate_norm(input,dim1))/n;
    x1= train_validate_norm(input,dim1):d:incorrect_L1_norm(input2,dim1);
    if(d==0)
        x1 = repmat(train_validate_norm(input,dim1),[1, n+1]);
    end
    d = (incorrect_L1_norm(input2,dim2)-train_validate_norm(input,dim2))/n;
    x2= train_validate_norm(input,dim2):d:incorrect_L1_norm(input2,dim2);
    if(d==0)
        x2 = repmat(train_validate_norm(input,dim2),[1, n+1]);
    end
scatter(x1,x2,4);

%plot corrected by L2.
h1(6) = scatter(incorrect_L1_norm(1,dim1),incorrect_L1_norm(1,dim2),incorrect_L1_norm(1,1),20);% failure case no. 5

%plot training data assigned by L2.
h1(7) = scatter(train_validate_norm(84,dim1),train_validate_norm(84,dim2),20);

%connect assigned training data to the corrected data (previous error data in L1)
input = 84;input2 =1;
    d = (incorrect_L1_norm(input2,dim1)-train_validate_norm(input,dim1))/n;
    x1= train_validate_norm(input,dim1):d:incorrect_L1_norm(input2,dim1);
    if(d==0)
        x1 = repmat(train_validate_norm(input,dim1),[1, n+1]);
    end
    d = (incorrect_L1_norm(input2,dim2)-train_validate_norm(input,dim2))/n;
    x2= train_validate_norm(input,dim2):d:incorrect_L1_norm(input2,dim2);
    if(d==0)
        x2 = repmat(train_validate_norm(input,dim2),[1, n+1]);
    end
scatter(x1,x2,4);

set(h1(1:3),'markersize',8);
set(h1(4),'marker','x');
set(h1(4),'SizeData',300);
set(h1(4),'linewidth',3.5);

set(h1(5),'marker','s');
set(h1(5),'SizeData',300);
set(h1(5),'linewidth',2);

set(h1(6),'marker','o');
set(h1(6),'SizeData',300);
set(h1(6),'linewidth',2);

set(h1(7),'marker','o');
set(h1(7),'SizeData',500);
set(h1(7),'linewidth',3);
%
legend(h1(1:7),{'Class 1','Class 2','Class 3',...
    'failure cases','assigned training data','corrected by L2','training data assigned by L2'...
    });
%   'Class 1 successful','Class 2 successful','Class 3 successful'});
xlabel(strcat('dimension ',int2str(dim1-1)));
ylabel(strcat('dimension ',int2str(dim2-1)));
title(strcat('L1 & L2 (Normalised) - Failure Test Data points in Dimension ',int2str(dim1-1),' and ',int2str(dim2-1)));
hold off;