% k = 10
%% class All cov k=10
dim1 = 2; 
dim2 = 3; 
trans = (G*(train_validate_norm(1:118,2:14))')';
%map idx_maha clusters to class labels
idx_maha_convert = zeros(118,1);
for i=1:118
    for j=1:size(class1,1)
        if(idx_maha(i) == class1(j))
           idx_maha_convert(i) = 1; 
           break
        end
    end
    for j=1:size(class2,1)
        if(idx_maha(i) == class2(j))
           idx_maha_convert(i) = 2; 
           break
        end
    end
    for j=1:size(class3,1)
        if(idx_maha(i) == class3(j))
           idx_maha_convert(i) = 3; 
           break
        end
    end
end
gscatter(trans(1:118,dim1),trans(1:118,dim2),idx_maha_convert);
hold on 

% class 1 [2,3,4]
h=scatter(cent_maha([2,3,4],dim1),cent_maha([2,3,4],dim2)); %all belong to class two
set(h,'marker','<');
set(h,'sizedata',200);
set(h,'linewidth',4);

%class 2 [1,5,7,9]
h=scatter(cent_maha([1,5,7,9],dim1),cent_maha([1,5,7,9],dim2)); %all belong to class two
set(h,'marker','>');
set(h,'sizedata',200);
set(h,'linewidth',4);

%class 3 [6,8,10]
h=scatter(cent_maha([6,8,10],dim1),cent_maha([6,8,10],dim2)); %all belong to class two
set(h,'marker','^');
set(h,'sizedata',200);
set(h,'linewidth',4);
title('k = 10 - Maha (with covariance matrix of all classes)');
xlabel(strcat('dimension ',int2str(dim1)));
ylabel(strcat('dimension ',int2str(dim2)));
set(gca,'fontsize',14);
legend('class 1','class 2','class 3',...
    'class 1 centers','class 3 centers','class 3 centers');
%%
figure;
gscatter(trans(:,dim1),trans(:,dim2),train_validate(1:118,1));

% cov all
% outliers 3,4 ,5,6 . 7,8
% 9 good seperation of all
% 11 good seperation of class 2..

%% class one cov k=10

dim1 = 1; 
dim2 = 2; 
trans = (G*normc(wine_data(:,2:14))')';
gscatter(trans(:,dim1),trans(:,dim2),wine_data(:,1));
hold on 

%class 1 [3]
h=scatter(cent_maha(3,dim1),cent_maha(3,dim2)); %all belong to class two
set(h,'marker','>');
set(h,'sizedata',200);
set(h,'linewidth',4);

%class 2 [1,5,7:9,10]
h=scatter(cent_maha([1,5,7:9,10],dim1),cent_maha([1,5,7:9,10],dim2)); %all belong to class two
set(h,'marker','<');
set(h,'sizedata',200);
set(h,'linewidth',4);

%class 3 [4,2,6]
h=scatter(cent_maha([4, 2, 6],dim1),cent_maha([4, 2, 6],dim2)); %all belong to class two
set(h,'marker','^');
set(h,'sizedata',200);
set(h,'linewidth',4);
title('k = 10 - Maha1 (with class one covariance matrix)');
xlabel(strcat('dimension ',int2str(dim1)));
ylabel(strcat('dimension ',int2str(dim2)));
set(gca,'fontsize',14);
legend('class 1','class 2','class 3',...
    'class 1 centers','class 3 centers','class 3 centers');
% dim 10 good seperation of class 1
%dim 2 and 7 good seperation of class 3
%dim 3 good seperation of class 3, some overlap with class 2 . 2& 3 good
%for showing the outlier centers

