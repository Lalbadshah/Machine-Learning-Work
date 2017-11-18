%loading training data

addpath(genpath('male'),genpath('female'));

Malefilelist = dir(sprintf('%s//male//train//*.jpg',pwd));
Femalefilelist = dir(sprintf('%s//female//train//*.jpg',pwd));

Male=[];
Female=[];
for k=1:length(Malefilelist)
    Male= [Male,reshape(imread(Malefilelist(k).name),[250*250,1])];
    Female= [Female,reshape(imread(Femalefilelist(k).name),[250*250,1])];
end
Train=[Male,Female];
clearvars Male Female;
%Mean training data values

Train_Avg = zeros(250*250,1); 

for c=1:250*250
    Train_Avg(c) = mean(Train(c,:)); 
end

save('Train_Avg.dat','Train_Avg','-ascii');

%Centering Data

Train = double(Train);
[TrainRow,TrainCol] = size(Train);

for i=1:TrainCol

    Train(:,i) = Train(:,i) - Train_Avg;             
    
end

%Covariance matrix

Train_Eig = transpose(Train)*Train;
[Train_Other_Eig_Vec,Train_Eig_Vals] = eig(Train_Eig);

%Best Eigen Vectors

Train_Cov_Eig_Face = Train*Train_Other_Eig_Vec;

%normalising

Train_Cov_Eig_Face = Train_Cov_Eig_Face/sqrt(Train_Eig_Vals);

Train_Cov_Eig_Face = Train_Cov_Eig_Face(:,end-199:end);

save('Training_Eigen_Faces.dat','Train_Cov_Eig_Face','-ascii');
%Function to project Data into vector space
part2();