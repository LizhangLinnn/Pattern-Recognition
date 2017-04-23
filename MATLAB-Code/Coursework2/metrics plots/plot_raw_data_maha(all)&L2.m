%% raw data
% %class one dim 1 & 2
figure;
dim1 = 5;
dim2 = 13;


dim1 = dim1 +1;
dim2 = dim2+1;

% ***************** training_ validation *****************


plot_cov_ClassOne = [cov_ClassOne(dim1-1,dim1-1),cov_ClassOne(dim1-1,dim2-1);...
    cov_ClassOne(dim2-1,dim1-1), cov_ClassOne(dim2-1,dim2-1)];
plot_cov_ClassTwo = [cov_ClassTwo(dim1-1,dim1-1),cov_ClassTwo(dim1-1,dim2-1);...
    cov_ClassTwo(dim2-1,dim1-1), cov_ClassTwo(dim2-1,dim2-1)];
plot_cov_ClassThree = [cov_ClassThree(dim1-1,dim1-1),cov_ClassThree(dim1-1,dim2-1);...
    cov_ClassThree(dim2-1,dim1-1), cov_ClassThree(dim2-1,dim2-1)];
plot_cov_All = [cov_All(dim1-1,dim1-1),cov_All(dim1-1,dim2-1);...
    cov_All(dim2-1,dim1-1), cov_All(dim2-1,dim2-1)];

%%

%mismatch maha
[n,d]=knnsearch(train_validate(1:118,2:14),test(:,2:14),'k',1,...
   'distance','mahalanobis');
mismatch_mahalanobis = [];
for i=1:40
   train_label = train_validate(n(i),1);
   if(train_label ~= test(i,1))
      mismatch_mahalanobis = horzcat(mismatch_mahalanobis,[i]); 
   end
end


%mismatch euclidean
[ncb,dcb]=knnsearch(train_validate(1:118,2:14),test(:,2:14),'k',1,...
   'distance','euclidean');
mismatch_euclidean = [];
for i=1:40
   train_label = train_validate(ncb(i),1);
   if(train_label ~= test(i,1))
      mismatch_euclidean = horzcat(mismatch_euclidean,[i]); 
   end
end

% mismatch L1
[ncb,dcb]=knnsearch(train_validate(1:118,2:14),test(:,2:14),'k',1,...
   'distance','minkowski','p',1);
mismatch_L1 = [];
for i=1:40
   train_label = train_validate(ncb(i),1);
   if(train_label ~= test(i,1))
      mismatch_L1 = horzcat(mismatch_L1,[i]); 
   end
end

%mismatch Chi-2
[~, mismatch_Chi2] = Chi2(train_validate(1:118,:), test);

%%
[n,d]=knnsearch(train_validate(1:118,2:14),test(mismatch_mahalanobis,2:14),'k',1,'distance','mahalanobis');
[ncb,dcb] = knnsearch(train_validate(1:118,2:14),test(mismatch_euclidean,2:14),'k',1,...
   'distance','euclidean');
gscatter(train_validate(1:118,dim1),train_validate(1:118,dim2),train_validate(1:118,1));

line(test(mismatch_mahalanobis(1),dim1),test(mismatch_mahalanobis(1),dim2),'marker','x','color','m',...
   'markersize',10,'linewidth',2,'linestyle','none')
% line(test(mismatch_euclidean,dim1),test(mismatch_euclidean,dim2),'marker','x','color','k',...
%    'markersize',10,'linewidth',2,'linestyle','none')
% line(test(mismatch_L1,dim1),test(mismatch_L1,dim2),'marker','o','color','c',...
%    'markersize',14,'linewidth',2,'linestyle','none')
% line(test(mismatch_Chi2,dim1),test(mismatch_Chi2,dim2),'marker','s','color','r',...
%    'markersize',20,'linewidth',2,'linestyle','none')


% line(train_validate(n,dim1),train_validate(n,dim2),'color',[.5 .5 .5],'marker','o',...
%    'linestyle','none','markersize',10)
% line(train_validate(ncb,dim1),train_validate(ncb,dim2),'color',[.5 .5 .5],'marker','p',...
%    'linestyle','none','markersize',10)
% connect mismatch1 to assigned train data
% for i=1:size(mismatch,2)
%     line([train_validate(n(i),dim1),test(mismatch(i),dim1)],[train_validate(n(i),dim2),test(mismatch(i),dim2)])
% end
legend('class one','class two','class three',...
     'Test sample');
hold on

h1 = error_ellipse(plot_cov_ClassOne,[test(mismatch_mahalanobis(1),dim1),test(mismatch_mahalanobis(1),dim2)],'conf',0.30);
h2 = error_ellipse(plot_cov_ClassTwo,[test(mismatch_mahalanobis(1),dim1),test(mismatch_mahalanobis(1),dim2)],'conf',0.30);
h3 = error_ellipse(plot_cov_ClassThree,[test(mismatch_mahalanobis(1),dim1),test(mismatch_mahalanobis(1),dim2)],'conf',0.30);
h4 = error_ellipse(plot_cov_All,[test(mismatch_mahalanobis(1),dim1),test(mismatch_mahalanobis(1),dim2)],'conf',0.30);

set(h1,'linewidth',2);
set(h2,'linewidth',2);
set(h3,'linewidth',2);
set(h4,'linewidth',2);

r=30;
ang=0:0.01:2*pi; 
xp=r*cos(ang);
yp=r*sin(ang);
h4 = plot(test(mismatch_mahalanobis(1),dim1)+xp,test(mismatch_mahalanobis(1),dim2)+yp);
set(h4,'linewidth',2);
legend('class one','class two','class three',...
     'Test sample',...
     'Maha Contour Weighted by Covariance of Class One Data',...
     'Maha Contour Weighted by Covariance of Class Two Data',...
     'Maha Contour Weighted by Covariance of Class Three Data',...
     'Maha Contour Weighted by Covariance of All Data',...
     'L2 Distance Contour');
title('Mahalanobis Distance with Covariance of Different Classes vs. L2 Distance');

% h = error_ellipse(plot_cov_ClassOne,[mean_ClassOne(dim1-1),mean_ClassOne(dim2-1)],'conf',0.80);
% h = error_ellipse(plot_cov_ClassTwo,[mean_ClassTwo(dim1-1),mean_ClassTwo(dim2-1)],'conf',0.80);
% h = error_ellipse(plot_cov_ClassThree,[mean_ClassThree(dim1-1),mean_ClassThree(dim2-1)],'conf',0.80);

hold off
%%
count = 0;
for i=1:40
    label_train = train_validate(n(i),1);
    if (label_train == test(i,1))
       count = count+1; 
    end
end
match = train_validate(n,1)-test(:,1); % 6 and 17 mismatch
accuracy_Maha = count/40;