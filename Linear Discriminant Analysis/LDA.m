%Loading training data
addpath(genpath('Test'),genpath('Train'));
Train=zeros(112*92,360);
l=1;
for k=1:40
    for i=1:9
        path = sprintf('%s//Train//s%d//%d.pgm',pwd,k,i);
        Train(:,l)= reshape(imread(path),[112*92,1]);
        l=l+1;
    end
end
size(diag(Train))

[Train_Coeffs] = pca(Train');
Train_PCA =zeros(300,360);

for c = 1:360
    Train_PCA(:,c) = Train_Coeffs(:,1:300)'*Train(:,c);
end
clearvars Train;
Train = Train_PCA;

% Within class Variance - Calculating SW
k=1;
Sw= zeros(300);

%Train = double(Train);
avg=zeros(300,40);

for i=1:40
    A = Train(:,k:9*i);
    k=k+9;
    avg(:,i) = mean(A,2); % mean of class
    Sk = (A-avg(:,i))*(A-avg(:,i))';
    Sw = Sw + Sk; 
end
SW =  Sw; 

% Inter Class variance - Calculating SB

Train_Avg = mean(Train,2);


SB = zeros(300);

for c=1:40
    SB = SB + 9*((avg(:,c)-Train_Avg)*(avg(:,c)-Train_Avg)');
end

V=[];
D=[];
[V, D]= eigs(SB,SW,40-1); 

plot(D,'X');
D = sort(diag(D),'descend');
%------------------------------Loading Test Data------------------------
Test=zeros(112*92,40);
n=1;
for m=1:40
    path2 = sprintf('%s\\Test\\s%d\\10.pgm',pwd,m);
    Test(:,n)= reshape(imread(path2),[112*92,1]);
    n=n+1;
end

Test_PCA =zeros(300,40);
for c = 1:40
    Test_PCA(:,c) = Train_Coeffs(:,1:300)'*Test(:,c);
end

Test = Test_PCA;

for c=1:40
    Test(:,c) = Test(:,c)-Train_Avg;
end

for c=1:360
    Train(:,c) = Train(:,c)-Train_Avg;
end

Proj_Train =[];
Proj_Train10 =[];
Proj_Train20 =[];
Proj_Train30 =[];
Proj_Test=[];
Proj_Test_10=[];
Proj_Test_20=[];
Proj_Test_30=[];

%--Projecting all Training images - USING Different Number of Eigen Vectors------------
for k=1:360
    Proj_Train = [Proj_Train,V'*Train(:,k)];
    Proj_Train10 = [Proj_Train10,V(:,1:10)'*Train(:,k)];
    Proj_Train20 = [Proj_Train20,V(:,1:20)'*Train(:,k)];
    Proj_Train30 = [Proj_Train30,V(:,1:30)'*Train(:,k)];
end
%------Projecting all Test images - USING Different Number of Eigen Vectors------------

for k=1:40
    Proj_Test = [Proj_Test,V'*Test(:,k)];
    Proj_Test_10 = [Proj_Test_10,V(:,1:10)'*Test(:,k)];
    Proj_Test_20 = [Proj_Test_20,V(:,1:20)'*Test(:,k)];
    Proj_Test_30 = [Proj_Test_30,V(:,1:30)'*Test(:,k)];
end
%--------------------Calculating the distances-----------------------------
Distances_39 = dist(Proj_Train,Proj_Test);
Distances_10 = dist(Proj_Train10,Proj_Test_10);
Distances_20 = dist(Proj_Train20,Proj_Test_20);
Distances_30 = dist(Proj_Train30,Proj_Test_30);

%--------------------Calculating the Accuracy-----------------------------

[~,I_39]=min(Distances_39);
[~,I_30]=min(Distances_30);
[~,I_20]=min(Distances_20);
[~,I_10]=min(Distances_10);

acc = [1,4];
acc(1,4) = accuracy(I_39);
acc(1,1) = accuracy(I_10);
acc(1,2) = accuracy(I_20);
acc(1,3) = accuracy(I_30);

figure;
plot(acc);
xticklabels({'10','','20','','30','','39'});
xlabel('No. of Dimensions');
ylabel('Accuracy %');



