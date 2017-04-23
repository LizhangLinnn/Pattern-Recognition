%% raw data
% %class one dim 1 & 2
figure;
dim1 = 5;
dim2 = 13;


dim1 = dim1 +1;
dim2 = dim2+1;

% ***************** training_ validation *****************


plot_cov_ClassOne_norm = [cov_ClassOne_norm(dim1-1,dim1-1),cov_ClassOne_norm(dim1-1,dim2-1);...
    cov_ClassOne_norm(dim2-1,dim1-1), cov_ClassOne_norm(dim2-1,dim2-1)];
plot_cov_ClassTwo_norm = [cov_ClassTwo_norm(dim1-1,dim1-1),cov_ClassTwo_norm(dim1-1,dim2-1);...
    cov_ClassTwo_norm(dim2-1,dim1-1), cov_ClassTwo_norm(dim2-1,dim2-1)];
plot_cov_ClassThree_norm = [cov_ClassThree_norm(dim1-1,dim1-1),cov_ClassThree_norm(dim1-1,dim2-1);...
    cov_ClassThree_norm(dim2-1,dim1-1), cov_ClassThree_norm(dim2-1,dim2-1)];
%%

%mismatch maha
[n,d]=knnsearch(train_validate_norm(1:118,2:14),test_norm(:,2:14),'k',1,...
   'distance','mahalanobis');
mismatch_mahalanobis = [];
for i=1:40
   train_label = train_validate_norm(n(i),1);
   if(train_label ~= test_norm(i,1))
      mismatch_mahalanobis = horzcat(mismatch_mahalanobis,[i]); 
   end
end


%mismatch euclidean
[ncb,dcb]=knnsearch(train_validate_norm(1:118,2:14),test_norm(:,2:14),'k',1,...
   'distance','euclidean');
mismatch_euclidean = [];
for i=1:40
   train_label = train_validate_norm(ncb(i),1);
   if(train_label ~= test_norm(i,1))
      mismatch_euclidean = horzcat(mismatch_euclidean,[i]); 
   end
end

% mismatch L1
[ncb,dcb]=knnsearch(train_validate_norm(1:118,2:14),test_norm(:,2:14),'k',1,...
   'distance','minkowski','p',1);
mismatch_L1 = [];
for i=1:40
   train_label = train_validate_norm(ncb(i),1);
   if(train_label ~= test_norm(i,1))
      mismatch_L1 = horzcat(mismatch_L1,[i]); 
   end
end

%mismatch Chi-2
[~, mismatch_Chi2] = Chi2(train_validate_norm(1:118,:), test_norm);

%%
[n,d]=knnsearch(train_validate_norm(1:118,2:14),test_norm(mismatch_L1,2:14),'k',1,'distance','minkowski','p',1);
[ncb,dcb] = knnsearch(train_validate_norm(1:118,2:14),test_norm(mismatch_euclidean,2:14),'k',1,...
   'distance','euclidean');
gscatter(train_validate_norm(1:118,dim1),train_validate_norm(1:118,dim2),train_validate_norm(1:118,1));

% line(test_norm(mismatch_maha,dim1),test_norm(mismatch_maha,dim2),'marker','s','color','m',...
%    'markersize',10,'linewidth',2,'linestyle','none')
line(test_norm(mismatch_euclidean,dim1),test_norm(mismatch_euclidean,dim2),'marker','x','color','k',...
   'markersize',10,'linewidth',2,'linestyle','none')
line(test_norm(mismatch_L1,dim1),test_norm(mismatch_L1,dim2),'marker','o','color','c',...
   'markersize',14,'linewidth',2,'linestyle','none')
% line(test_norm(mismatch_Chi2,dim1),test_norm(mismatch_Chi2,dim2),'marker','s','color','r',...
%    'markersize',20,'linewidth',2,'linestyle','none')


% line(train_validate_norm(n,dim1),train_validate_norm(n,dim2),'color',[.5 .5 .5],'marker','s',...
%    'linestyle','none','markersize',14)
% line(train_validate_norm(ncb,dim1),train_validate_norm(ncb,dim2),'color',[.5 .5 .5],'marker','p',...
%    'linestyle','none','markersize',14)
% connect mismatch1 to assigned train data
% for i=1:size(mismatch,2)
%     line([train_validate_norm(n(i),dim1),test_norm(mismatch(i),dim1)],[train_validate_norm(n(i),dim2),test_norm(mismatch(i),dim2)])
% end
legend('class one','class two','class three',...
     'Incorrect Prediction (L2)',...
    'Incorrect Prediction (L1)');
hold on

% h = error_ellipse(plot_cov_ClassOne_norm,[mean_ClassOne_norm(dim1-1),mean_ClassOne_norm(dim2-1)],'conf',0.9);
% h = error_ellipse(plot_cov_ClassTwo_norm,[mean_ClassTwo_norm(dim1-1),mean_ClassTwo_norm(dim2-1)],'conf',0.9);
% h = error_ellipse(plot_cov_ClassThree_norm,[mean_ClassThree_norm(dim1-1),mean_ClassThree_norm(dim2-1)],'conf',0.9);

hold off
%%
count = 0;
for i=1:40
    label_train = train_validate_norm(n(i),1);
    if (label_train == test_norm(i,1))
       count = count+1; 
    end
end
match = train_validate_norm(n,1)-test_norm(:,1); % 6 and 17 mismatch
accuracy_Maha = count/40;