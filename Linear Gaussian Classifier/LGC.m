function [accs,acci] = LGC(X)
addpath(genpath('io'));

train_x = loadMNISTImages(sprintf('%s\\data\\train-images-idx3-ubyte',pwd));
train_y = loadMNISTLabels(sprintf('%s\\data\\train-labels-idx1-ubyte',pwd))';
test_x = loadMNISTImages(sprintf('%s\\data\\t10k-images-idx3-ubyte',pwd));
test_y = loadMNISTLabels(sprintf('%s\\data\\t10k-labels-idx1-ubyte',pwd))';

[train_coeffs] = pca(train_x');
Train_PCA =zeros(X,size(train_x,2));

Train_avg = mean(train_x,2);

%Centering and Projecting Test and train data
for c = 1:size(train_x,2)
    train_x(:,c) = train_x(:,c) - Train_avg;
    Train_PCA(:,c) = train_coeffs(:,1:X)'*train_x(:,c);
end
Test_PCA = zeros(X,size(test_x,2));
for c = 1:size(test_x,2)
    test_x(:,c) = test_x(:,c) - Train_avg;
    Test_PCA(:,c) = train_coeffs(:,1:X)'*test_x(:,c);
end
%Mean and Covariance of each class
class_means = zeros(X,length(unique(train_y)));
class_counts = zeros(1,length(unique(train_y)));
for i = 1:size(train_y,2)
    class_counts(:,train_y(:,i)+1) = class_counts(:,train_y(:,i)+1)+1;
    class_means(:,train_y(:,i)+1)=class_means(:,train_y(:,i)+1)+Train_PCA(:,i);
end
for i = 1:size(class_counts,2)
    class_means(:,i) = class_means(:,i)./class_counts(:,i);
end
class_covs = zeros(X,X*size(class_counts,2));
for j=1:size(class_counts,2)
    tempcov = [];
    for k=1:size(train_y,2)
        if(j==train_y(:,k)+1)
            tempcov = [tempcov,Train_PCA(:,k)];
        end
    end
    class_covs(:,(X*(j-1))+1:X*j)= cov(tempcov');
end

% Weighted average of covariances - Shared Covariance
shared_cov=zeros(X,X);
for c=1:size(class_counts,2)
    shared_cov = shared_cov + class_counts(:,c)*class_covs(:,(X*(c-1))+1:X*c);
end

shared_cov = shared_cov/size(train_y,2);

%Multivariate Normal Probability density function - and testing


mvds = [];
test_results=[];
for m=1:size(class_counts,2)
    mvds = (class_counts(:,m)/size(train_y,2))* mvnpdf(Test_PCA',class_means(:,m)',shared_cov);
    test_results = [test_results,mvds];
end

[~,test_results] = max(test_results,[],2);
test_results = (test_results-1)';

%Accuracy
acc = 0;
for i=1:size(test_y,2)
    if(test_results(i) == test_y(i))
       acc = acc+1;
    end
end
accs = acc/size(test_y,2)*100;

% Now calculating using individual covariance matrices

mvds = [];
test_results=[];
for m=1:size(class_counts,2)
    mvds = (class_counts(:,m)/size(train_y,2))* mvnpdf(Test_PCA',class_means(:,m)',class_covs(:,(X*(m-1))+1:X*m));
    test_results = [test_results,mvds];
end

[~,test_results] = max(test_results,[],2);
test_results = (test_results-1)';

%Accuracy
acc = 0;
for i=1:size(test_y,2)
    if(test_results(i) == test_y(i))
       acc = acc+1;
    end
end
acci = acc/size(test_y,2)*100;
end


