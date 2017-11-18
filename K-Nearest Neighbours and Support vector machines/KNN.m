function [Results]= KNN(Data,k)

Test_Matrix = Data(:,3869:end)';
Train_Data = Data(:,1:3868)';
neighborIndices = zeros(size(Test_Matrix,1),k);

Data_size = size(Train_Data,1);
Test_size = size(Test_Matrix,1);
for i=1:Test_size
    dist = sum((repmat(Test_Matrix(i,:),Data_size,1)-Train_Data).^2,2);
    [~,positions] = sort(dist,'ascend');
    neighborIndices(i,:) = positions(1:k);
end
%Male faces will have index between 1 & 1934 and for Female indeces are
%between - 1935 & 3868
for j=1:Test_size
    M = nnz(neighborIndices(j,:)<1934);
    F = k-M;
    if(M>F)
        Results(j,1)=1;
    else
        Results(j,1)=-1;    
    end
end
end
