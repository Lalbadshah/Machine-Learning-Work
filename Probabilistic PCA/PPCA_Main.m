
accuracy_individual=[];
for eig = [50, 100, 150,200]
    aci=PPCA(eig);
    accuracy_individual=[accuracy_individual,aci];
end
figure;
X = [50, 100, 150,200];
plot(X,accuracy_individual,'g--x');

title('Linear Gaussian Classification accuracy')
xlabel('No. Of Eigen Vectors used')
ylabel('Accuracy %')

legend('Using Individual Covariance')
figure;
bar(X,accuracy_individual)
xlabel('No. Of Eigen Vectors used')
ylabel('Accuracy %')
