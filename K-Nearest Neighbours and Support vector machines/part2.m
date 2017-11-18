function [] = part2()
%Function to project Data into vector space

%Transforming Training and Testing Data


%Loading Data 

addpath(genpath('male'),genpath('female'));

Malefilelist = dir(sprintf('%s//male//train//*.jpg',pwd));
Femalefilelist = dir(sprintf('%s//female//train//*.jpg',pwd));

Malefilelist1 = dir(sprintf('%s//male//test//*.jpg',pwd));
Femalefilelist1 = dir(sprintf('%s//female//test//*.jpg',pwd));

Male=[];
Male1=[];
Female=[];
Female1=[];
for k=1:length(Malefilelist)
    Male= [Male,reshape(imread(Malefilelist(k).name),[250*250,1])];
    Female= [Female,reshape(imread(Femalefilelist(k).name),[250*250,1])];
end
for k=1:length(Malefilelist1)
    Male1= [Male1,reshape(imread(Malefilelist1(k).name),[250*250,1])];
    Female1= [Female1,reshape(imread(Femalefilelist1(k).name),[250*250,1])];
end
Train=[Male,Female,Male1,Female1];
clearvars Male Female Male1 Female1;
Eig_faces = load('Training_Eigen_Faces.dat');
Avg = load('Train_Avg.dat');

%centering training data
[~,TrainCol] = size(Train);
Train = double(Train);
for i=1:TrainCol

    Train(:,i) = Train(:,i) - Avg;             
    
end

%Projecting the images onto the eigen vector space

Weight_Vectors = [];
weight = zeros(200,1);

for i=1:5868
    for j=1:200
        weight(j,1) = Eig_faces(:,j)'*Train(:,i);
    end
    Weight_Vectors = [Weight_Vectors,weight];
end

save('Weight_Vector_Projections.dat','Weight_Vectors','-ascii');



ends