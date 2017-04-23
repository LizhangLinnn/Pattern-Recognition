% k =3
%% class All cov k=3
dim1 = 4; 
dim2 = 3; 
trans = (G*normc(wine_data(:,2:14))')';
gscatter(trans(:,dim1),trans(:,dim2),wine_data(:,1));
hold on

% class 1 [3]
h=scatter(cent_maha(2,dim1),cent_maha(2,dim2)); %all belong to class two
set(h,'marker','<');
set(h,'sizedata',200);
set(h,'linewidth',4);

%class 2 [1]
h=scatter(cent_maha(3,dim1),cent_maha(3,dim2)); %all belong to class two
set(h,'marker','<');
set(h,'sizedata',200);
set(h,'linewidth',4);

%class 3 [2]
h=scatter(cent_maha(1,dim1),cent_maha(1,dim2)); %all belong to class two
set(h,'marker','^');
set(h,'sizedata',200);
set(h,'linewidth',4);
title('k = 3 - Maha (with covariance matrix of all classes)');
xlabel(strcat('dimension ',int2str(dim1)));
ylabel(strcat('dimension ',int2str(dim2)));
set(gca,'fontsize',14);
legend('class 1','class 2','class 3',...
    'class 1 centers','class 2 centers','class 3 centers');

% cov all
% outliers 3,4 ,5,6 . 7,8
% 9 good seperation of all
% 11 good seperation of class 2..

%% class one cov k=3
subplot(121);
%original
dim1 = 4; 
dim2 = 3; 

trans = (G*normc(train_validate(:,2:14))')';

trans = (G*normc(train_validate(1:118,2:14))')';
idx_maha_convert = train_validate(1:118,1);
for i=1:118
    if(idx_maha_convert(i) == 1)
        idx_maha_convert(i) = class1(1,1);
    elseif(idx_maha_convert(i) == 2)
        idx_maha_convert(i) = class2(1,1);
    elseif(idx_maha_convert(i) == 3)
        idx_maha_convert(i) = class3(1,1);

    end
end

gscatter(trans(:,dim1),trans(:,dim2),idx_maha_convert);

title('data distribution after Maha1 (with class one covariance matrix)');
xlabel(strcat('dimension ',int2str(dim1)));
ylabel(strcat('dimension ',int2str(dim2)));
set(gca,'fontsize',14);
legend('class 1','class 2','class 3',...
    'class 1 centers','class 2 centers','class 3 centers');



% ****************************************************
% kmeans clustering
subplot(122);
gscatter(trans(:,dim1),trans(:,dim2),idx_maha);
hold on 


%class 1 [3]
h=scatter(cent_maha(1,dim1),cent_maha(1,dim2)); %all belong to class two
set(h,'marker','<');
set(h,'sizedata',200);
set(h,'linewidth',4);

%class 2 
h=scatter(cent_maha(2,dim1),cent_maha(2,dim2)); %all belong to class two
set(h,'marker','^');
set(h,'sizedata',200);
set(h,'linewidth',4);

%class 3 
h=scatter(cent_maha(3,dim1),cent_maha(3,dim2)); %all belong to class two
set(h,'marker','>');
set(h,'sizedata',200);
set(h,'linewidth',4);


title('k = 3 - kmeans clustering');
xlabel(strcat('dimension ',int2str(dim1)));
ylabel(strcat('dimension ',int2str(dim2)));
set(gca,'fontsize',14);
legend('class 1','class 2','class 3',...
    'class 1 centers','class 2 centers','class 3 centers');
% dim 10 good seperation of class 1
%dim 2 and 7 good seperation of class 3
%dim 3 good seperation of class 3, some overlap with class 2 . 2& 3 good
%for showing the outlier centers

