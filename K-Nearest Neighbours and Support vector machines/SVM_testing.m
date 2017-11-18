Data = load('Weight_Vector_Projections.dat');

classMatrix = [ones(1,1934)*-1,ones(1,1934)];

%50 bases
projection = Data(end-49:end,1:3868)';
testing = Data(end-49:end,3869:end)';


 SVM = fitcsvm(projection,classMatrix,'KernelFunction','gaussian','Solver','SMO','Standardize',true);
 Result50= predict(SVM,testing);
 
%100 bases

projection = Data(end-99:end,1:3868)';
testing = Data(end-99:end,3869:end)';


 SVM = fitcsvm(projection,classMatrix,'KernelFunction','gaussian','Solver','SMO','Standardize',true);
 Result100= predict(SVM,testing);
 

%200 bases

projection = Data(:,1:3868)';
testing = Data(:,3869:end)';


 SVM = fitcsvm(projection,classMatrix,'KernelFunction','gaussian','Solver','SMO','Standardize',true);
 
 Result= predict(SVM,testing);

%Calculating accuracy
 
Results_Truth = ones(2000,1);
Results_Truth(1:1000,1) = Results_Truth(1:1000,1).*-1;

Accuracy=[];

Accuracy(1) = ceil((1 - (nnz(Result50+Results_Truth)/2000))*100);
Accuracy(2) = ceil((1 - (nnz(Result100+Results_Truth)/2000))*100);
Accuracy(3) = ceil((1 - (nnz(Result+Results_Truth)/2000))*100);


 figure;

plot([50,100,200],Accuracy);
xlabel("Number of bases");
ylabel("Accuracy%");