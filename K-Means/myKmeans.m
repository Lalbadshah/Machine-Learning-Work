
function [C,I] = myKmeans(X,K,max_iter)
%
Inputs:
X - Input image
K - number of centroids
max_iter - maximum number of iterations
outputs:
C - list of centroids
I - Cluster assignment
%
cents=zeros(K,3);
[r,c,p] = size(X);
centroid = [];
for k = 1:K
    cents(k,1) = randi(r);
    cents(k,2) = randi(c);
    centroid(k,1)= X(cents(k,1),cents(k,2),1);
    centroid(k,2)= X(cents(k,1),cents(k,2),2);
    centroid(k,3)= X(cents(k,1),cents(k,2),3);
end

C=centroid;
D=C;
X = reshape(X,r*c,p);
X = double(X);
I = [];
for iter = 1:max_iter
    for i=1:r*c
        dist=[];
            for k=1:K
                dist = [dist, norm(C(k,:) - X(i,:), 1)];
            end
            [~,Cluster_assignment]=min(dist);
            I(i) = Cluster_assignment;
    end
   sums = zeros(K,4);
   for l=1:r
       for o=1:K
        if(I(l)==o)
            sums(o,1:3) = sums(o,1:3) + X(l,:);
            sums(o,4) = sums(o,4)+1;
        end
       end
   end
   for n=1:K
       D(n,:) = sums(n,1:3)./sums(n,4);
   end
   if(C==D)
       break
   else
       C = D;
   end

end
