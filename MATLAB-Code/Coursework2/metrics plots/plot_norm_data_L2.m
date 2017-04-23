%% normalised
% %class one dim 1 & 2
figure;
dim1 = 2;
dim2 = 13;


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
%L2
% failure - only class 2
h1(4) = scatter(incorrect_L2_norm(:,dim1),incorrect_L2_norm(:,dim2),[],'g');% failure case no. 5
% success - three classes
%h1(5:7) = gscatter(correct_L2_norm(:,dim1),correct_L2_norm(:,dim2),correct_L2_norm(:,1)); %success case no. 1
% zoom in
% limits = [1.88, 3.26,390, 873];
% axis(limits)

set(h1(1:3),'markersize',8);
set(h1(4),'marker','x');
set(h1(4),'SizeData',300);
set(h1(4),'linewidth',2);
%
legend(h1(1:4),{'Class 1','Class 2','Class 3',...
    'Class 2 failure'...
    });
%   'Class 1 successful','Class 2 successful','Class 3 successful'});
xlabel(strcat('dimension ',int2str(dim1-1)));
ylabel(strcat('dimension ',int2str(dim2-1)));
title(strcat('L2 (Normalised) - Failure Test Data points in Dimension ',int2str(dim1-1),' and ',int2str(dim2-1)));
hold off;
