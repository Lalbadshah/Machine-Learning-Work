Data = load('Weight_Vector_Projections.dat');



Results_Truth = ones(2000,1);
Results_Truth(1:1000,1) = Results_Truth(1:1000,1).*-1;

Results_1=KNN(Data,1);
Results_5=KNN(Data,5);
Results_10=KNN(Data,10);
Results_50=KNN(Data,50);
Results_100=KNN(Data,100);

% Calculating accuracy

Accuracy_1 = ceil((1 - (nnz(Results_1+Results_Truth)/2000))*100);
Accuracy_5 = ceil((1 - (nnz(Results_5+Results_Truth)/2000))*100);
Accuracy_10 = ceil((1 - (nnz(Results_10+Results_Truth)/2000))*100);
Accuracy_50 = ceil((1 - (nnz(Results_50+Results_Truth)/2000))*100);
Accuracy_100 = ceil((1 - (nnz(Results_100+Results_Truth)/2000))*100);

Accuracy = [Accuracy_1,Accuracy_5,Accuracy_10,Accuracy_50,Accuracy_100];

figure;
plot(Accuracy);
xticklabels({'1','','5','','10','','50','','100'});
xlabel("Number of Neighbors");
ylabel("Accuracy%");

for c=99:-50:49
Results_1=KNN(Data(end-c:end,:),1);
Results_5=KNN(Data(end-c:end,:),5);
Results_10=KNN(Data(end-c:end,:),10);
Results_50=KNN(Data(end-c:end,:),50);
Results_100=KNN(Data(end-c:end,:),100);

% Calculating accuracy

Accuracy_1 = ceil((1 - (nnz(Results_1+Results_Truth)/2000))*100);
Accuracy_5 = ceil((1 - (nnz(Results_5+Results_Truth)/2000))*100);
Accuracy_10 = ceil((1 - (nnz(Results_10+Results_Truth)/2000))*100);
Accuracy_50 = ceil((1 - (nnz(Results_50+Results_Truth)/2000))*100);
Accuracy_100 = ceil((1 - (nnz(Results_100+Results_Truth)/2000))*100);

Accuracy = [Accuracy_1,Accuracy_5,Accuracy_10,Accuracy_50,Accuracy_100];

figure;
plot(Accuracy);
xticklabels({'1','','5','','10','','50','','100'});
xlabel("Number of Neighbors");
ylabel("Accuracy%");

end