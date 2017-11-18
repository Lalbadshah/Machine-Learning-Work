function [acci] = PPCA(k)
%Load data%
train_x = loadMNISTImages(sprintf('%s\\data\\train-images-idx3-ubyte',pwd));
train_y = loadMNISTLabels(sprintf('%s\\data\\train-labels-idx1-ubyte',pwd))';
test_x = loadMNISTImages(sprintf('%s\\data\\t10k-images-idx3-ubyte',pwd));
test_y = loadMNISTLabels(sprintf('%s\\data\\t10k-labels-idx1-ubyte',pwd))';
w = ones(size(train_x,1),k)*rand();
sigma = rand();

%Cetering Data
Train_avg = mean(train_x,2);
for c = 1:size(train_x,2)
    train_x(:,c) = train_x(:,c) - Train_avg;
end
%Sample covariance
S = sum(sum(train_x.^2));
D = 784;
N=60000;
M=w'*w+sigma*eye(k);
V = inv(chol(M));
Minv = V*V';
lgl=[];
count=0;
mx = 20;
while(count<=mx)
    %Calculating Log Liklihood
    wx = w'*train_x;
    lnm = 2*sum(log(diag(chol(M))))+ (D-k)*log(sigma);
    x = (-N/2)*(D*log(2*22/7)+lnm+trace(Minv*S));
    lgl = [lgl,x(1)];

    count=count+1;
    % E step
    M=w'*w+sigma*eye(k);
    V = inv(chol(M));
    Minv = V*V';
    Ez=[];
    Ezn=[];

    Ez = Minv*w'*train_x;
    Ezn = sigma*Minv+Ez*Ez';

    % M Step
    w = (train_x*Ez')/(Ez*Ez'+sigma*Minv);
    sigdash = (sum(sum(train_x.^2)) - 2*sum(sum(Ez.*(w'*train_x))) + sum(trace(Ezn*(w'*w))))/(N*D);
    sigma = sigdash;
end

Train_PCA =zeros(k,size(train_x,2));

%Centering and Projecting Test and train data
for c = 1:size(train_x,2)
    Train_PCA(:,c) = w(:,1:k)'*train_x(:,c);
end
Test_PCA = zeros(k,size(test_x,2));
for c = 1:size(test_x,2)
    test_x(:,c) = test_x(:,c) - Train_avg;
    Test_PCA(:,c) = w(:,1:k)'*test_x(:,c);
end
%Mean and Covariance of each class
class_means = zeros(k,length(unique(train_y)));
class_counts = zeros(1,length(unique(train_y)));
for i = 1:size(train_y,2)
    class_counts(:,train_y(:,i)+1) = class_counts(:,train_y(:,i)+1)+1;
    class_means(:,train_y(:,i)+1)=class_means(:,train_y(:,i)+1)+Train_PCA(:,i);
end
for i = 1:size(class_counts,2)
    class_means(:,i) = class_means(:,i)./class_counts(:,i);
end
class_covs = zeros(k,k*size(class_counts,2));
for j=1:size(class_counts,2)
    tempcov = [];
    for m=1:size(train_y,2)
        if(j==train_y(:,m)+1)
            tempcov = [tempcov,Train_PCA(:,m)];
        end
    end
    class_covs(:,(k*(j-1))+1:k*j)= cov(tempcov');
end


% Now calculating using individual covariance matrices

mvds = [];
test_results=[];
for m=1:size(class_counts,2)
    mvds = (class_counts(:,m)/size(train_y,2))* mvnpdf(Test_PCA',class_means(:,m)',class_covs(:,(k*(m-1))+1:k*m));
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
