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
%%

%mismatch Chi-2
[~, mismatch_hist] = histogram_intersection_norm(train_validate(1:118,:), test);

%%


gscatter(train_validate(1:118,dim1),train_validate(1:118,dim2),train_validate(1:118,1));


line(test(mismatch_hist,dim1),test(mismatch_hist,dim2),'marker','x','color','k',...
   'markersize',20,'linewidth',2,'linestyle','none')


% line(train_validate(n,dim1),train_validate(n,dim2),'color',[.5 .5 .5],'marker','s',...
%    'linestyle','none','markersize',14)
% line(train_validate(ncb,dim1),train_validate(ncb,dim2),'color',[.5 .5 .5],'marker','p',...
%    'linestyle','none','markersize',14)
% connect mismatch1 to assigned train data
% for i=1:size(mismatch,2)
%     line([train_validate(n(i),dim1),test(mismatch(i),dim1)],[train_validate(n(i),dim2),test(mismatch(i),dim2)])
% end
legend('class one','class two','class three',...
     'Incorrect Prediction (hist)');
hold on

h = error_ellipse(plot_cov_ClassOne,[mean_ClassOne(dim1-1),mean_ClassOne(dim2-1)],'conf',0.9);
h = error_ellipse(plot_cov_ClassTwo,[mean_ClassTwo(dim1-1),mean_ClassTwo(dim2-1)],'conf',0.9);
h = error_ellipse(plot_cov_ClassThree,[mean_ClassThree(dim1-1),mean_ClassThree(dim2-1)],'conf',0.9);

hold off
