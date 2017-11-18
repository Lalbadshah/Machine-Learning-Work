
accuracy_shared =[];
accuracy_individual=[];
for eig = [50, 100, 200, 300, 400, 500]
    [acs,aci]=LGC(eig);
    accuracy_shared=[accuracy_shared,acs];
    accuracy_individual=[accuracy_individual,aci];
end
figure;
X = [50, 100, 200, 300, 400, 500];
plot(X,accuracy_shared,'b--o',X,accuracy_individual,'g--x');

title('Linear Gaussian Classification accuracy')
xlabel('No. Of Eigen Vectors used')
ylabel('Accuracy %')

legend('Using Shared Covariance','Using Individual Covariance')