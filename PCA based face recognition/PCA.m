%PCA based face recognition
%Prateek Bhatnagar
%-------------------------------Instructions - Incase Everything doesnt work automatically-------------------------------
% 1.Add Folders Test and Train To path(with subfolders)
% 2.pwd should work automatically for the file paths - however if there is
% any trouble just replace %s(In 'path' and 'path2') with the location where Main_PCA.m is located
%
addpath(genpath('Test'),genpath('Train'));
X=zeros(112*92,360);
l=1;
%converting to vector from image matrix
for k=1:40
    path = sprintf('%s\\Train\\s%d\\',pwd,k);
    files = dir(fullfile(path,'*.pgm'));
    for i=1:length(files)
        X(:,l)= reshape(imread(files(i).name),[112*92,1]);
        l=l+1;
    end
end

%Creating a Matrix of the vectors
S1 = X;
%Finding the mean of the vectors
S1_Avg = zeros(112*92,1);
for c=1:112*92
     S1_Avg(c) = mean(S1(c,:));
end
S1 = double(S1);
%Normalising the vectors - i.e. Centering the data
for i=1:360
    S1(:,i) = S1(:,i) - S1_Avg;
end
%Covariance matrix
S1_Eig = transpose(S1)*S1;
[S1_Other_Eig_Vec,S1_Eig_Vals] = eig(S1_Eig);
%Best Eigen Vectors
S1_Cov_Eig_Face = S1*S1_Other_Eig_Vec;
%normalising
S1_Cov_Eig_Face = S1_Cov_Eig_Face/sqrt(S1_Eig_Vals);
%Eigen Value Plot
figure(1);
plot(S1_Eig_Vals,'o');
title('Plot of Eigen Values');
figure(2);
imshow(reshape(S1_Avg,[112,92]),[]);
title('Mean Face');
%First 3 and last eigen faces
figure(3);
subplot(4,1,1);
imshow(reshape(S1_Cov_Eig_Face(:,end),[112,92]),[]);
title('First 3 eigen faces');
subplot(4,1,2);
imshow(reshape(S1_Cov_Eig_Face(:,end-1),[112,92]),[]);
subplot(4,1,3);
imshow(reshape(S1_Cov_Eig_Face(:,end-2),[112,92]),[]);
subplot(4,1,4);
imshow(reshape(S1_Cov_Eig_Face(:,end-39),[112,92]),[]);
title('Last eigen face');
%Slicing 40 relevant(Non zero) Eigen Faces
%-----------------------------IMPORTANT NOTE------------------------------
% The order of the eigen faces is in Ascending order
E_faces = S1_Cov_Eig_Face(:,end-39:end);
%Using 10,20,30,40 vectors for projection
Proj_10 = E_faces(:,1:10)'*S1(:,1);
Proj_20 = E_faces(:,1:20)'*S1(:,1);
Proj_30 = E_faces(:,1:30)'*S1(:,1);
Proj_40(:,1) = E_faces'*S1(:,1);
figure(4);
subplot(5,1,1);
imshow(reshape(S1(:,1)+S1_Avg,[112,92]),[]);
title('Original');
subplot(5,1,2);
imshow(reshape((E_faces(:,1:10)*Proj_10)+S1_Avg,[112,92]),[]);
title('10 Vectors');
subplot(5,1,3);
imshow(reshape((E_faces(:,1:20)*Proj_20)+S1_Avg,[112,92]),[]);
title('20 Vectors');
subplot(5,1,4);
imshow(reshape((E_faces(:,1:30)*Proj_30)+S1_Avg,[112,92]),[]);
title('30 Vectors');
% In the last case the most important eigen face is added
% - since for the given code the order of the eigen faces is in ascending order - hence the
% resulting image is almost identical to the original
% However in the previous cases the eigen faces were not significant enough
% to reconstruct the original image
subplot(5,1,5);
imshow(reshape((E_faces*Proj_40(:,1))+S1_Avg,[112,92]),[]);
title('40 Vectors');
%--------------------------------Weight Vectors for entire Training set----
for o=1:40
    Proj_40(:,o) = E_faces'*S1(:,o);
    Proj_30(:,o) = E_faces(:,1:30)'*S1(:,o);
    Proj_20(:,o) = E_faces(:,1:20)'*S1(:,o);
    Proj_10(:,o) = E_faces(:,1:10)'*S1(:,o);
end
%--------------------------------Testing-----------------------------------

Test=zeros(112*92,40);
%Importing test data as image vectors
n=1;
for m=1:40
    path2 = sprintf('%s\\Test\\s%d\\10.pgm',pwd,m);
    Test(:,n)= reshape(imread(path2),[112*92,1]);
    n=n+1;
end

%Centering test data
Test_Avg = zeros(112*92,1);
for c=1:112*92
     Test_Avg(c) = mean(Test(c,:));
end
Test = double(Test);
for i=1:40
    Test(:,i) = Test(:,i) - Test_Avg;
end

for k=1:40
    Test_Proj_10(:,k)= E_faces(:,1:10)'*Test(:,k);
    Test_Proj_20(:,k)= E_faces(:,1:20)'*Test(:,k);
    Test_Proj_30(:,k)= E_faces(:,1:30)'*Test(:,k);
    Test_Proj_40(:,k)= E_faces'*Test(:,k);
end
figure(5);
imshow(reshape((E_faces*Test_Proj_40(:,1))+Test_Avg,[112,92]),[]);
title('40 Vectors - Projection - CLOSE IMAGE of S1 ');

% Calculating Distance between Test Weight vectors and Trained weight
% vectors - Euclidian Distance
for p=1:40
    for q=1:40
        Distance_40(q) = sqrt(sum((Test_Proj_40(:,p) - Proj_40(:,q)) .^ 2));
        Distance_30(q) = sqrt(sum((Test_Proj_30(:,p) - Proj_30(:,q)) .^ 2));
        Distance_20(q) = sqrt(sum((Test_Proj_20(:,p) - Proj_20(:,q)) .^ 2));
        Distance_10(q) = sqrt(sum((Test_Proj_10(:,p) - Proj_10(:,q)) .^ 2));
    end
    [Min_Value,Min_Index(p,4)] = min(Distance_40);
    [~,Min_Index(p,3)] = min(Distance_30);
    [~,Min_Index(p,2)] = min(Distance_20);
    [~,Min_Index(p,1)] = min(Distance_10);
end
%---------IMPORTANT NOTE--------
%Min_Index holds the approximate classifications made by the classifier
%based on the category - each of the columns is a respective category for
%number of vectors used i.e. Column 1 is for 10 vectors, 2 for 20 Vectors and so
%on... Each row position represents an entry from the testing data
f = figure(6);
colnames = {'10-Category','20-Category','30-Category','40-Category'};
t=uitable(f,'Data',Min_Index,'ColumnName',colnames,'Position',[20 30 350 350]);
