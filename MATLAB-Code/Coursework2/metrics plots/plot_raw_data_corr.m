%% raw data
% %class one dim 1 & 2
figure;
dim1 = 1;
dim2 = 2;


dim1 = dim1 +1;
dim2 = dim2+1;

% ***************** training_ validation *****************


plot_cov_ClassOne = [cov_ClassOne(dim1-1,dim1-1),cov_ClassOne(dim1-1,dim2-1);...
    cov_ClassOne(dim2-1,dim1-1), cov_ClassOne(dim2-1,dim2-1)];
plot_cov_ClassTwo = [cov_ClassTwo(dim1-1,dim1-1),cov_ClassTwo(dim1-1,dim2-1);...
    cov_ClassTwo(dim2-1,dim1-1), cov_ClassTwo(dim2-1,dim2-1)];
plot_cov_ClassThree = [cov_ClassThree(dim1-1,dim1-1),cov_ClassThree(dim1-1,dim2-1);...
    cov_ClassThree(dim2-1,dim1-1), cov_ClassThree(dim2-1,dim2-1)];

% mismatch Corr
[ncb,dcb]=knnsearch(train_validate(1:118,2:14),test(:,2:14),'k',1,...
   'distance','correlation');
mismatch_Corr = [];
for i=1:40
   train_label = train_validate(ncb(i),1);
   if(train_label ~= test(i,1))
      mismatch_Corr = horzcat(mismatch_Corr,[i]); 
   end
end


[n,d]=knnsearch(train_validate(1:118,2:14),test(mismatch_Corr,2:14),'k',1,'distance','correlation');
% [ncb,dcb] = knnsearch(train_validate(1:118,2:14),test(mismatch_euclidean,2:14),'k',1,...
%    'distance','euclidean');
gscatter(train_validate(1:118,dim1),train_validate(1:118,dim2),train_validate(1:118,1));

% line(test(mismatch_maha,dim1),test(mismatch_maha,dim2),'marker','s','color','m',...
%    'markersize',10,'linewidth',2,'linestyle','none')
line(test(mismatch_Corr,dim1),test(mismatch_Corr,dim2),'marker','x','color','k',...
   'markersize',10,'linewidth',2,'linestyle','none')
% line(test(mismatch_L1,dim1),test(mismatch_L1,dim2),'marker','o','color','c',...
%    'markersize',14,'linewidth',2,'linestyle','none')
% line(test(mismatch_Chi2,dim1),test(mismatch_Chi2,dim2),'marker','s','color','r',...
%    'markersize',20,'linewidth',2,'linestyle','none')


line(train_validate(n,dim1),train_validate(n,dim2),'color',[.5 .5 .5],'marker','o',...
   'linestyle','none','markersize',10)
% line(train_validate(ncb,dim1),train_validate(ncb,dim2),'color',[.5 .5 .5],'marker','p',...
%    'linestyle','none','markersize',10)
% connect mismatch1 to assigned train data
for i=1:size(mismatch_Corr,2)
    line([train_validate(n(i),dim1),test(mismatch_Corr(i),dim1)],[train_validate(n(i),dim2),test(mismatch_Corr(i),dim2)])
end
legend('class one','class two','class three',...
     'Incorrect Prediction (L2)',...
    'Assigned Training Data');
hold on

h = error_ellipse(plot_cov_ClassOne,[mean_ClassOne(dim1-1),mean_ClassOne(dim2-1)],'conf',0.80);
h = error_ellipse(plot_cov_ClassTwo,[mean_ClassTwo(dim1-1),mean_ClassTwo(dim2-1)],'conf',0.80);
h = error_ellipse(plot_cov_ClassThree,[mean_ClassThree(dim1-1),mean_ClassThree(dim2-1)],'conf',0.80);

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